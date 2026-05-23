import os
import json
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import shutil
import random
import numpy as np
SEED=42
random.seed(SEED)
np.random.seed(SEED)
# ========================
# 配置区
# ========================
MODEL_NAME = "/vla/users/luojingzhou/data/checkpoints/hf-models/Qwen/Qwen3-8B"
USE_4BIT = False  # 若显存不足，设为 True 启用 4-bit 量化（需 bitsandbytes）

# ========================
# 初始化模型
# ========================
print("Loading Qwen3-8B model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if USE_4BIT:
    from transformers import BitsAndBytesConfig
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

model.eval()

# 特殊 token ID：用于截断 thinking 区域
THINK_END_ID = 151668  # </think> 的 token ID

def generate_with_thinking(prompt: str, max_new_tokens: int = 1024) -> str:
    """
    使用 Qwen3 的 thinking 模式生成回答，并返回最终 content（不含 thinking）
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # 不启用 thinking 模式
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    input_len = model_inputs.input_ids.shape[1]
    output_ids = generated_ids[0][input_len:].tolist()

    # 尝试定位 </think> 的位置
    try:
        # 从后往前找第一个 THINK_END_ID
        rev_index = output_ids[::-1].index(THINK_END_ID)
        index = len(output_ids) - rev_index
    except ValueError:
        index = 0  # 没有找到 </think>，则认为无 thinking 区域

    # 提取最终内容（跳过 thinking 部分）
    content_ids = output_ids[index:]
    content = tokenizer.decode(content_ids, skip_special_tokens=True).strip("\n ")

    return content

def generate_variants(original_task: str, num_return_sequences: int = 5) -> list[str]:
    """
    使用 Qwen3-8B 生成多种风格的同义指令。
    返回包含原始任务 + 生成变体的列表（去重后，确保原始任务在第一位）
    """
    if original_task in ["", " "]:
        return [original_task]
    prompts = [
        # 1. User Intent (Non-imperative, natural desire)
        f'Rewrite the robotic command "{original_task}" as a natural expression of user intent, without using imperative mood or "please". Output only the rewritten sentence.',

        # 2. Functional Description (What, not how)
        f'Rephrase "{original_task}" as a functional goal that describes what should be achieved, not how to do it. Output only the rewritten sentence.',

        # 3. Functional Reference (Refer to objects by their purpose, not name)
        f'Rewrite "{original_task}" by referring to objects based on their function or typical use instead of their names, do not mention the object name. Output only the rewritten sentence.',

        # 4. Polite / Courteous Request
        f'Rewrite "{original_task}" as a polite and courteous request a human might naturally say. Output only the rewritten sentence.',

        # 5. Concise Command (Minimal but clear)
        f'Make "{original_task}" as concise as possible while keeping all essential actions and objects. Output only the rewritten sentence.',

        # 6. Teaching/Instructional Style
        f'Explain the task "{original_task}" clearly as if teaching a new robot learner, focusing on purpose and clarity. Output only the rewritten sentence.',

        # 7. Abstract Goal
        f'Summarize the core objective of "{original_task}" at a high level, ignoring low-level details. Output only the rewritten sentence.'
    ]

    variants = [original_task]  # 原始任务始终保留

    for prompt in prompts:
        for _ in range(num_return_sequences):
            try:
                generated = generate_with_thinking(prompt, max_new_tokens=512)
                
                # 清理结尾标点
                generated = generated.rstrip(' .,;:!?。！？；：')
                generated = generated.replace('The user wants', 'I want')
                # 过滤无效结果
                if (
                    generated and
                    len(generated) >= 8 and
                    generated != original_task and
                    not generated.lower().startswith(("sorry", "i cannot", "i don't", "unable", "error", "i'm sorry"))
                ):
                    variants.append(generated)
            except Exception as e:
                print(f"Generation error for prompt '{prompt[:50]}...': {e}")
                continue

    # 去重但保持顺序
    seen = set()
    unique_variants = []
    for v in variants:
        if v not in seen:
            unique_variants.append(v)
            seen.add(v)

    print(f"Original task: {original_task} -> Generated {len(unique_variants)-1} variants.-> {unique_variants}")
    return unique_variants

# ========================
# 主流程：查找文件并处理
# ========================
def main():
    # task_path = '/vla/users/luojingzhou/data/datasets/libero_plus_lerobot_4suite/lerobot/*/meta/tasks.jsonl'
    # task_path = '/vla/users/luojingzhou/data/datasets/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/*/meta/tasks.jsonl'
    task_path = '/vla/users/luojingzhou/data/datasets/libero_lerobot/*/meta/tasks.jsonl'
    task_files = glob.glob(task_path)
    # task_files = ['/vla/users/luojingzhou/data/datasets/VLA-Dagger/GR00TN-1.6-InterVL3.5-3B-liberoplus_all_6w/liberoplus_all_lerobotv2_multirobotstate/meta/tasks.jsonl']
    print(f"Found {len(task_files)} task files to process.")

    for task_file in task_files:
        input_path = Path(task_file)
        output_path = input_path.parent / "tasks.jsonl"
        print(f"\nProcessing: {input_path} → {output_path}")
        if os.path.exists(str(input_path).replace('.jsonl','_raw.jsonl')):
            print(f"  Output file already exists, skipping.")
            continue
        output_lines = []
        with open(input_path, 'r', encoding='utf-8') as f_in:
            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    task_index = data["task_index"]
                    original_task = data["task"]
                    print(f"  Line {line_num}: Generating variants for task '{original_task[:200]}...'")

                    task_list = generate_variants(original_task)

                    new_entry = {
                        "task_index": task_index,
                        "task": original_task,
                        "task_list": task_list
                    }
                    output_lines.append(json.dumps(new_entry, ensure_ascii=False))

                except Exception as e:
                    print(f"  Error on line {line_num}: {e}")
                    continue
        # 复制原文件
        shutil.copy2(str(input_path), str(input_path).replace('.jsonl','_raw.jsonl'))
        
        # 写入新文件
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line in output_lines:
                f_out.write(line + '\n')

        print(f"  Saved {len(output_lines)} entries to {output_path}")

    print("\n✅ All done!")

if __name__ == "__main__":
    main()
