import os
import json
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import shutil
import random
import numpy as np
import ray
from ray.util.queue import Queue
import argparse
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ========================
# 全局配置（可按需修改）
# ========================
MODEL_NAME = "Qwen3/Qwen3-8B"
USE_4BIT = False
TASK_PATH_PATTERN = '/data0/luojingzhou/data/datasets/hcp_panda_real_data_easy_lerobotv2/*/meta/tasks.jsonl'
THINK_END_ID = 151668


# ========================
# Ray Actor: 每个 GPU 一个 Worker，模型只加载一次
# ========================
@ray.remote(num_gpus=1)
class ModelWorker:
    def __init__(self, task_queue, result_queue):
        self.task_queue = task_queue
        self.result_queue = result_queue

        print(f"[Actor on GPU {ray.get_gpu_ids()}] Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

        if USE_4BIT:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        self.model.eval()
        print(f"[Actor on GPU {ray.get_gpu_ids()}] Model loaded.")

    def generate_with_thinking(self, prompt: str, max_new_tokens: int = 1024) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        input_len = model_inputs.input_ids.shape[1]
        output_ids = generated_ids[0][input_len:].tolist()

        try:
            rev_index = output_ids[::-1].index(THINK_END_ID)
            index = len(output_ids) - rev_index
        except ValueError:
            index = 0

        content_ids = output_ids[index:]
        content = self.tokenizer.decode(content_ids, skip_special_tokens=True).strip("\n ")
        return content

    def generate_variants(self, original_task: str, num_return_sequences: int = 5) -> list[str]:
        if not original_task.strip():
            return [original_task]
        prompts = [
            # 1. User Intent (Non-imperative, natural desire)
            f'Rewrite the robotic command "{original_task}" as a natural expression of user intent, without using imperative mood or "please". Output only the rewritten synonym sentence.',

            # 2. Functional Description (What, not how)
            f'Rephrase "{original_task}" as a functional goal that describes what should be achieved, not how to do it. Output only the rewritten synonym sentence.',

            # 3. Functional Reference (Refer to objects by their purpose, not name)
            f'Rewrite "{original_task}" by referring to objects based on their function, appearance or typical use instead of their names. Do not mention the object name, but be able to accurately associate the target object with its description. Output only the rewritten synonym sentence.',

            # 4. Polite / Courteous Request
            f'Rewrite "{original_task}" as a polite and courteous request a human might naturally say. Output only the rewritten synonym sentence.',

            # 5. Concise Command (Minimal but clear)
            f'Make "{original_task}" as concise as possible while keeping all essential actions and objects. Output only the rewritten synonym sentence.',

            # 6. Teaching/Instructional Style
            f'Explain the task "{original_task}" clearly as if teaching a new robot learner, focusing on purpose and clarity. Output only the rewritten synonym sentence.',

            # 7. Abstract Goal
            f'Summarize the core objective of "{original_task}" at a high level, ignoring low-level details that do not change the task execution order. Output only the rewritten synonym sentence.'
        ]

        variants = [original_task]
        for prompt in prompts:
            for _ in range(num_return_sequences):
                try:
                    generated = self.generate_with_thinking(prompt, max_new_tokens=512)
                    generated = generated.rstrip(' .,;:!?。！？；：')
                    generated = generated.replace('The user wants', 'I want')
                    if "." in original_task[-2:]:
                        generated = generated+'.'
                    elif "?" in original_task[-2:]:
                        generated = generated+'?'
                    elif "!" in original_task[-2:]:
                        generated = generated+'!'
                    if (
                        generated and
                        len(generated) >= 8 and
                        generated != original_task and
                        not generated.lower().startswith(("sorry", "i cannot", "i don't", "unable", "error", "i'm sorry"))
                    ):
                        variants.append(generated)
                except Exception as e:
                    continue

        seen = set()
        unique = []
        for v in variants:
            if v not in seen:
                unique.append(v)
                seen.add(v)
        return unique

    def process_file(self, task_file: str):
        input_path = Path(task_file)
        if os.path.exists(str(input_path).replace('.jsonl', '_raw.jsonl')):
            return "skipped"

        output_lines = []
        try:
            with open(input_path, 'r', encoding='utf-8') as f_in:
                for line_num, line in enumerate(f_in, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        original_task = data["task"]
                        task_list = self.generate_variants(original_task)
                        new_entry = {
                            "task_index": data["task_index"],
                            "task": original_task,
                            "task_list": task_list
                        }
                        output_lines.append(json.dumps(new_entry, ensure_ascii=False))
                    except Exception as e:
                        print(f"[{task_file}] Line {line_num} error: {e}")
                        continue

            # 保存备份和新文件
            shutil.copy2(str(input_path), str(input_path).replace('.jsonl', '_raw.jsonl'))
            with open(input_path.parent / "tasks.jsonl", 'w', encoding='utf-8') as f_out:
                for line in output_lines:
                    f_out.write(line + '\n')

            return "success"
        except Exception as e:
            print(f"[{task_file}] FAILED: {e}")
            return "failed"

    def run(self):
        """持续从队列中取任务，直到收到 None（哨兵值）"""
        while True:
            try:
                task_file = self.task_queue.get(block=True, timeout=1.0)
                if task_file is None:  # 哨兵，表示结束
                    break
                status = self.process_file(task_file)
                self.result_queue.put((task_file, status))
            except Exception as e:
                # 超时或异常，继续尝试
                continue


# ========================
# 主函数：使用队列分发动态任务
# ========================
def main():
    argparse.ArgumentParser(description="Generate task variants using Ray")
    parser.add_argument("--task_path_pattern", default="/data0/luojingzhou/data/datasets/hcp_panda_real_data_easy_lerobotv2/*/meta/tasks.jsonl", help="Pattern to match task files")
    args = parser.parse_args()
    args.task_path_pattern
    ray.init(ignore_reinit_error=True)

    num_gpus = int(ray.cluster_resources().get("GPU", 0))
    if num_gpus == 0:
        raise RuntimeError("No GPU available!")

    # 获取所有待处理文件
    task_files = glob.glob(args.task_path_pattern)
    print(f"Found {len(task_files)} task files.")
    if not task_files:
        print("No files to process. Exiting.")
        return

    # 创建共享队列
    task_queue = Queue(maxsize=len(task_files) + num_gpus)  # +num_gpus 用于哨兵
    result_queue = Queue()

    # 将所有任务放入队列
    for f in task_files:
        task_queue.put(f)

    # 添加哨兵（每个 worker 一个）
    for _ in range(num_gpus):
        task_queue.put(None)

    # 启动 workers
    workers = [ModelWorker.remote(task_queue, result_queue) for _ in range(num_gpus)]
    worker_refs = [worker.run.remote() for worker in workers]

    # === 动态收集结果 ===
    results = []
    expected_results = len(task_files)
    print(f"Waiting for {expected_results} results...")

    # 使用一个计数器，避免死等
    while len(results) < expected_results:
        try:
            # 设置较短超时（如 60 秒），避免永久阻塞
            res = result_queue.get(timeout=60)
            results.append(res)
            print(f"\rProgress: {len(results)}/{expected_results}", end="", flush=True)
        except Exception as e:
            # 超时可能是正常现象（比如某个 worker 还在跑大任务）
            # 但我们不能无限等，需检查是否所有 workers 都已退出
            print(f"\n[Warning] Timeout waiting for result. Current: {len(results)}/{expected_results}")

            # 检查 workers 是否全部完成（非阻塞）
            ready, not_ready = ray.wait(worker_refs, timeout=0.1)
            if len(ready) == len(worker_refs):
                print("\nAll workers finished, but some results missing.")
                break
            else:
                # 还有 worker 在运行，继续等待
                continue

    print(f"\nCollected {len(results)} / {expected_results} results.")

    # 等待所有 workers 真正退出（可选，用于确保资源释放）
    try:
        ray.get(worker_refs, timeout=30)  # 最多等 30 秒
    except Exception as e:
        print(f"Some workers failed to exit cleanly: {e}")

    # 统计
    success = sum(1 for _, s in results if s == "success")
    skipped = sum(1 for _, s in results if s == "skipped")
    failed = sum(1 for _, s in results if s == "failed")

    print("\n" + "="*60)
    print(f"✅ All done! Total files: {len(results)}")
    print(f"   Success: {success}")
    print(f"   Skipped: {skipped}")
    print(f"   Failed : {failed}")
    print("="*60)


if __name__ == "__main__":
    main()
