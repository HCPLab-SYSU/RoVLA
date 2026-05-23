---
license: apache-2.0
pipeline_tag: image-text-to-text
library_name: transformers
base_model:
  - OpenGVLab/InternVL3_5-4B-MPO
base_model_relation: finetune
datasets:
  - OpenGVLab/MMPR-v1.2
  - OpenGVLab/MMPR-Tiny
language:
  - multilingual
tags:
  - internvl
  - custom_code
---

# InternVL3_5-4B

[\[📂 GitHub\]](https://github.com/OpenGVLab/InternVL)  [\[📜 InternVL 1.0\]](https://huggingface.co/papers/2312.14238)  [\[📜 InternVL 1.5\]](https://huggingface.co/papers/2404.16821)  [\[📜 InternVL 2.5\]](https://huggingface.co/papers/2412.05271)  [\[📜 InternVL2.5-MPO\]](https://huggingface.co/papers/2411.10442)  [\[📜 InternVL3\]](https://huggingface.co/papers/2504.10479) [\[📜 InternVL3.5\]](https://huggingface.co/papers/2508.18265)

[\[🆕 Blog\]](https://internvl.github.io/blog/)  [\[🗨️ Chat Demo\]](https://chat.intern-ai.org.cn/)  [\[🚀 Quick Start\]](#quick-start)  [\[📖 Documents\]](https://internvl.readthedocs.io/en/latest/)

<div align="center">
  <img width="500" alt="image" src="https://cdn-uploads.huggingface.co/production/uploads/64006c09330a45b03605bba3/zJsd2hqd3EevgXo6fNgC-.png">
</div>

## Introduction

We introduce *InternVL3.5*, a new family of open-source multimodal models that significantly advances versatility, reasoning capability, and inference efficiency along the InternVL series. A key innovation is the *Cascade Reinforcement Learning (Cascade RL)* framework, which enhances reasoning through a two-stage process: offline RL for stable convergence and online RL for refined alignment. This coarse-to-fine training strategy leads to substantial improvements on downstream reasoning tasks, e.g., MMMU and MathVista. To optimize efficiency, we propose a *Visual Resolution Router (ViR)* that dynamically adjusts the resolution of visual tokens without compromising performance. Coupled with ViR, our Decoupled *Vision-Language Deployment (DvD)* strategy separates the vision encoder and language model across different GPUs, effectively balancing computational load. These contributions collectively enable InternVL3.5 to achieve up to a +16.0\% gain in overall reasoning performance and a 4.05 \\(\times\\) inference speedup compared to its predecessor, i.e., InternVL3. In addition, InternVL3.5 supports novel capabilities such as GUI interaction and embodied agency. Notably, our largest model, i.e.,  InternVL3.5-241B-A28B, attains state-of-the-art results among open-source MLLMs across general multimodal, reasoning, text, and agentic tasks—narrowing the performance gap with leading commercial models like GPT-5. All models and code are publicly released.

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/performance.jpg)

> Hatched bars represent closed-source commercial models. We report average scores on a set of multimodal general, reasoning, text, and agentic benchmarks: MMBench v1.1 (en), MMStar,BLINK, HallusionBench, AI2D, OCRBench, MMVet, MME-RealWorld (en), MVBench, VideoMME, MMMU, MathVista, MathVision, MathVerse, DynaMath, WeMath, LogicVista, MATH500, AIME24, AIME25, GPQA, MMLU-Pro, GAOKAO, IFEval, SGP-Bench, VSI-Bench, ERQA, SpaCE-10, and OmniSpatial.

See [quick start](#quick-start) for how to use our model.

## InternVL3.5 Family

In the following table, we provide an overview of the InternVL3.5 series.
To maintain consistency with earlier generations, we provide two model formats: [the GitHub format](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B), consistent with prior releases, and [the HF format](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B-HF), aligned with the official Transformers standard.

> If you want to convert the checkpoint between these two formats, please refer to the scripts about [custom2hf](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/tools/internvl_custom2hf.py) and [hf2custom](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/tools/internvl_hf2custom.py).


### Github Format


| Model                 | #Vision Param | #Language Param | #Total Param | HF Link                                                                        | ModelScope Link                                                                          |
| --------------------- | ------------- | --------------- | ------------ | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| InternVL3.5-1B        | 0.3B          | 0.8B            | 1.1B         | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-1B)                      | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-1B)                      |
| InternVL3.5-2B        | 0.3B          | 2.0B            | 2.3B         | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-2B)                      | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-2B)                      |
| InternVL3.5-4B        | 0.3B          | 4.4B            | 4.7B         | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-4B)                      | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-4B)                      |
| InternVL3.5-8B        | 0.3B          | 8.2B            | 8.5B         | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-8B)                      | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-8B)                      |
| InternVL3.5-14B       | 0.3B          | 14.8B           | 15.1B        | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-14B)                     | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-14B)                     |
| InternVL3.5-38B       | 5.5B          | 32.8B           | 38.4B        | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-38B)                     | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-38B)                     |
| InternVL3.5-20B-A4B   | 0.3B          | 20.9B           | 21.2B-A4B    | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview) | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview) |
| InternVL3.5-30B-A3B   | 0.3B          | 30.5B           | 30.8B-A3B    | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B)                 | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-30B-A3B)                 |
| InternVL3.5-241B-A28B | 5.5B          | 235.1B          | 240.7B-A28B  | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B)               | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-241B-A28B)               |


### HuggingFace Format


| Model                    | #Vision Param | #Language Param | #Total Param | HF Link                                                                           | ModelScope Link                                                                             |
| ------------------------ | ------------- | --------------- | ------------ | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| InternVL3.5-1B-HF        | 0.3B          | 0.8B            | 1.1B         | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF)                      | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-1B-HF)                      |
| InternVL3.5-2B-HF        | 0.3B          | 2.0B            | 2.3B         | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF)                      | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-2B-HF)                      |
| InternVL3.5-4B-HF        | 0.3B          | 4.4B            | 4.7B         | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-4B-HF)                      | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-4B-HF)                      |
| InternVL3.5-8B-HF        | 0.3B          | 8.2B            | 8.5B         | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF)                      | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-8B-HF)                      |
| InternVL3.5-14B-HF       | 0.3B          | 14.8B           | 15.1B        | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-14B-HF)                     | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-14B-HF)                     |
| InternVL3.5-38B-HF       | 5.5B          | 32.8B           | 38.4B        | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-38B-HF)                     | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-38B-HF)                     |
| InternVL3.5-20B-A4B-HF   | 0.3B          | 20.9B           | 21.2B-A4B    | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview-HF) | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview-HF) |
| InternVL3.5-30B-A3B-HF   | 0.3B          | 30.5B           | 30.8B-A3B    | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B-HF)                 | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-30B-A3B-HF)                 |
| InternVL3.5-241B-A28B-HF | 5.5B          | 235.1B          | 240.7B-A28B  | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B-HF)               | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-241B-A28B-HF)               |


![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/performance_overall.jpg)

> We conduct the evaluation with [VLMEvalkit](https://github.com/open-compass/VLMEvalKit). ***To enable the Thinking mode of our model, please set the system prompt to [R1_SYSTEM_PROMPT](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/vlm/internvl/internvl_chat.py#L38).*** When enabling Thinking mode, we recommend setting `do_sample=True` and `temperature=0.6` to mitigate undesired repetition.

Our training pipeline comprises four stages: Multimodal Continual Pre-Training (**CPT**), Supervised Fine-Tuning (**SFT**), and Cascade Reinforcement Learning (**CascadeRL**). In CascadeRL, we first fine-tune the model using Mixed Preference Optimization (**MPO**) under an offline RL setting, followed by **GSPO** under an oneline RL setting.
For the Flash version of InternVL3.5, we additionally introduce a lightweight training stage, termed Visual Consistency Learning (**ViCO**), which reduces the token cost required to represent an image patch.

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/training_pipeline.jpg)

Here, we also open-source the model weights after different training stages for potential research usage.
***If you're unsure which version to use, please select the one without any suffix, as it has completed the full training pipeline.***


| Model                            | Training Pipeline     | HF Link                                                                     | ModelScope Link                                                                       |
| -------------------------------- | --------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| InternVL3.5-1B-Pretrained        | CPT                   | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-1B-Pretrained)        | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-1B-Pretrained)        |
| InternVL3.5-1B-Instruct          | CPT + SFT             | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-1B-Instruct)          | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-1B-Instruct)          |
| InternVL3.5-1B-MPO               | CPT + SFT + MPO       | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-1B-MPO)               | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-1B-MPO)               |
| InternVL3.5-1B                   | CPT + SFT + CascadeRL | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-1B)                   | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-1B)                   |
| InternVL3.5-2B-Pretrained        | CPT                   | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-2B-Pretrained)        | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-2B-Pretrained)        |
| InternVL3.5-2B-Instruct          | CPT + SFT             | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-2B-Instruct)          | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-2B-Instruct)          |
| InternVL3.5-2B-MPO               | CPT + SFT + MPO       | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-2B-MPO)               | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-2B-MPO)               |
| InternVL3.5-2B                   | CPT + SFT + CascadeRL | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-2B)                   | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-2B)                   |
| InternVL3.5-4B-Pretrained        | CPT                   | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-4B-Pretrained)        | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-4B-Pretrained)        |
| InternVL3.5-4B-Instruct          | CPT + SFT             | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-4B-Instruct)          | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-4B-Instruct)          |
| InternVL3.5-4B-MPO               | CPT + SFT + MPO       | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-4B-MPO)               | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-4B-MPO)               |
| InternVL3.5-4B                   | CPT + SFT + CascadeRL | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-4B)                   | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-4B)                   |
| InternVL3.5-8B-Pretrained        | CPT                   | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-8B-Pretrained)        | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-8B-Pretrained)        |
| InternVL3.5-8B-Instruct          | CPT + SFT             | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-8B-Instruct)          | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-8B-Instruct)          |
| InternVL3.5-8B-MPO               | CPT + SFT + MPO       | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-8B-MPO)               | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-8B-MPO)               |
| InternVL3.5-8B                   | CPT + SFT + CascadeRL | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-8B)                   | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-8B)                   |
| InternVL3.5-14B-Pretrained       | CPT                   | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-14B-Pretrained)       | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-14B-Pretrained)       |
| InternVL3.5-14B-Instruct         | CPT + SFT             | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-14B-Instruct)         | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-14B-Instruct)         |
| InternVL3.5-14B-MPO              | CPT + SFT + MPO       | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-14B-MPO)              | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-14B-MPO)              |
| InternVL3.5-14B                  | CPT + SFT + CascadeRL | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-14B)                  | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-14B)                  |
| InternVL3.5-30B-A3B-Pretrained   | CPT                   | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B-Pretrained)   | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-30B-A3B-Pretrained)   |
| InternVL3.5-30B-A3B-Instruct     | CPT + SFT             | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B-Instruct)     | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-30B-A3B-Instruct)     |
| InternVL3.5-30B-A3B-MPO          | CPT + SFT + MPO       | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B-MPO)          | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-30B-A3B-MPO)          |
| InternVL3.5-30B-A3B              | CPT + SFT + CascadeRL | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B)              | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-30B-A3B)              |
| InternVL3.5-38B-Pretrained       | CPT                   | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-38B-Pretrained)       | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-38B-Pretrained)       |
| InternVL3.5-38B-Instruct         | CPT + SFT             | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-38B-Instruct)         | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-38B-Instruct)         |
| InternVL3.5-38B-MPO              | CPT + SFT + MPO       | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-38B-MPO)              | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-38B-MPO)              |
| InternVL3.5-38B                  | CPT + SFT + CascadeRL | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-38B)                  | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-38B)                  |
| InternVL3.5-241B-A28B-Pretrained | CPT                   | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B-Pretrained) | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-241B-A28B-Pretrained) |
| InternVL3.5-241B-A28B-Instruct   | CPT + SFT             | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B-Instruct)   | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-241B-A28B-Instruct)   |
| InternVL3.5-241B-A28B-MPO        | CPT + SFT + MPO       | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B-MPO)        | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-241B-A28B-MPO)        |
| InternVL3.5-241B-A28B            | CPT + SFT + CascadeRL | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B)            | [🤖 link](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-241B-A28B)            |


The Flash version of our model will be released as soon as possible.



## Model Architecture

`InternVL3.5`:
This series of models follow the "ViT–MLP–LLM" paradigm adopted in previous versions of InternVL.
We initialize the language model using the Qwen3 series and GPT-OSS, and the vision encoder using InternViT-300M and InternViT-6B.
The Dynamic High Resolution strategy introduced in InternVL1.5 is also retained in our design.


`InternVL3.5-Flash`:
Compared to InternVL3.5, InternVL3.5-Flash further integrates the *Visual Resolution Router (ViR)*, thus yielding a series of  efficient variants friendly  suitable for  resource-constrained scenarios. 
Specifically, in InternVL3.5, each image patch is initially represented as 1024 visual tokens for the vision encoder, which are then compressed into 256 tokens via a pixel shuffle module before being passed to the Large Language Model (LLM).
In InternVL3.5-Flash, as shown in the Figure below, an additional pixel shuffle module with a higher compression rate is included, enabling the compression of visual tokens down to 64 tokens.
For each patch, the patch router determines the appropriate compression rate by assessing its semantic richness, and routes it to the corresponding pixel shuffle module accordingly.
Benefiting from this patch-aware compression mechanism, InternVL3.5-Flash is able to reduce the number of visual tokens by 50\% while maintaining nearly 100\% of the performance of InternVL3.5.


![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/architecture.jpg)

## Training and Deployment Strategy

### Pre-Training

During the pre-training stage, we update all model parameters jointly using the combination of large-scale text and multimodal corpora. Specifically, given an arbitrary training sample consisting of a multimodal token sequence \\(\mathbf{x}=\left(x_1, x_2, \ldots, x_L\right)\\), the next token prediction (NTP) loss is calculated on each text token as follows:

$$
    \mathcal{L}_{i}=-\log p_\theta\left(x_i \mid x_1, \ldots, x_{i-1}\right),
$$

where \\(x_i\\) is the predicted token and  prefix tokens in \\(\{x_1, x_2, \ldots, x_{i-1}\}\\) can be either  text tokens or  image tokens. Notably, for conversation samples, only response tokens  are included for the calculation of the loss.
Additionally, to mitigate bias toward either longer or shorter responses during training, we adopt the square averaging to re-weight the NTP loss  as follows:

$$
\mathcal{L}_{i}^{'} = \frac{w_i}{\sum_j w_j} \cdot \mathcal{L}_i, \quad w_i = \frac{1}{N^{0.5}},
$$

where \\(N\\) denotes the number of tokens in the training sample on which the loss needs to be calculated. The random JPEG compression is also included to enhance the model's real-world performance.

### Supervised Fine-Tuning

During the SFT phase, we adopt the same objective as in the pre-training stage and use the  square-root averaging strategy to calculate the final loss.  In this stage, the context window is set to 32K tokens to adapt long-context information.
Compared to InternVL3, the SFT stage of InternVL3.5 contains  more high-quality and  diverse training data derived from three sources: 

(1) Instruction-following data from InternVL3, which are reused to preserve broad coverage of vision–language tasks. 

(2) Multimodal reasoning data in the "Thinking" mode, which are included to instill long-thinking capabilities in the model. To construct such data, we first use InternVL3-78B to describe the image and then input the description into DeepSeek-R1 to sample rollouts with detailed reasoning processes. Rollouts with an incorrect final answer are filtered out. The questions in these datasets cover various expert domains, such as mathematics and scientific disciplines, thereby strengthening performance on different reasoning tasks. 

(3) Capability-expansion datasets, which endow InternVL3.5 with new skills, including GUI-based interaction, embodied interaction, and scalable vect

### Cascade Reinforcement Learning

Cascade RL aims to combine the benefits of offline RL and online RL to progressively facilitate the post-training of MLLMs in an efficient manner.
Specifically, we first fine-tune the model using an offline RL algorithm as an efficient warm-up stage to reach a satisfied results, which can guarantee the high-quality rollouts for the latter stage. 
Subsequently, we employ an online RL algorithm to further refine the output distribution based on rollouts generated by the model itself.  Compared to the single offline or online RL stage, our cascaded RL achieves significant performance improvements at a fraction of the GPU time cost.



During the offline RL stage, we employ mixed preference optimization (MPO) to fine-tune the model. Specifically, the training objective of MPO is a combination of preference loss \\(\mathcal{L}_{p}\\), quality loss \\(\mathcal{L}_{q}\\), and generation loss \\(\mathcal{L}_{g}\\), which can be formulated as follows:

$$
    \mathcal{L}_{\text{MPO}}=
    w_{p} \mathcal{L}_{p}
    +
    w_{q} \mathcal{L}_{q}
    +
    w_{g} \mathcal{L}_{g}
    ,
$$

where \\(w_{*}\\) represents the weight assigned to each loss component.
The DPO loss, BCO loss, and LM loss serve as the preference loss, quality loss, and generation loss, respectively.


During the online RL stage, we employ GSPO, without reference model constraints, as our online RL algorithm, which we find more effective in training both dense and mixture-of-experts (MoE) models. Similar to GRPO, the advantage is defined as the normalized reward across responses sampled from the same query.
The training objective of GSPO is given by:

$$
    \mathcal{L}_{\mathrm{GSPO}}(\theta)=\mathbb{E}_{x \sim \mathcal{D},\left\{y_i\right\}_{i=1}^G \sim \pi_{\theta \text { old }}(\cdot \mid x)}\left[\frac{1}{G} \sum_{i=1}^G \min \left(s_i(\theta) \widehat{A}_i, \operatorname{clip}\left(s_i(\theta), 1-\varepsilon, 1+\varepsilon\right) \widehat{A}_i\right)\right],
$$

where the importance sampling ratio is defined as the geometric mean of the per-token ratios.

> Please see [our paper](https://huggingface.co/papers/2508.18265) for more technical and experimental details.


### Visual Consistency Learning


We further include ViCO as an additional training stage to integrate the *visual resolution router (ViR)* into InternVL3.5, thereby reducing the inference cost of InternVL3.5. The obtained efficient version of InternVL3.5 are termed as *InternVL3.5-Flash*. In particular, ViCO comprises two stages:

`Consistency training`:
In this stage, the entire model is trained to minimize the divergence between response distributions conditioned on visual tokens with different compression rates.
In practice, we introduce an extra reference model, which is frozen and initialized with InternVL3.5.
Given a sample, each image patch is represented as either 256 or 64 tokens, and the training objective is defined as follows:


$$
\mathcal{L}_\text{ViCO} =
\mathbb{E}_{\xi \sim \mathcal{R}} \Bigg[
\frac{1}{N} \sum_{i=1}^{N} \mathrm{KL} \Big(
\pi_{\theta_{ref}}\left(y_i \mid y_{<i}, I\right) \;\Big\|\;
\pi_{\theta_{policy}}\left(y_i \mid y_{<i}, I_\xi\right)
\Big)
\Bigg],
$$

where \\(\mathrm{KL}\) denotes the KL divergence and \(\xi\) denotes the compression rate, which is uniformly sampled from \(\{\frac{1}{4},\frac{1}{16}\}\). The image \(I_\xi\) is represented as 256 tokens when \(\xi=\frac{1}{4}\) and 64 tokens when \(\xi=\frac{1}{16}\). Notably, the reference model always performs inference with \(\xi=\frac{1}{4}\).


`Router training`:
This stage aims to train the ViR to select an appropriate trade-off resolution for different inputs.
ViR is formulated as a binary classifier and trained using standard cross-entropy loss.
To construct the route targets, we first compute the KL divergence between the model outputs conditioned on uncompressed visual tokens (i.e., 256 tokens per patch) and those conditioned on compressed visual tokens (i.e., 64 tokens per patch).
During this stage, the main MLLM (ViT, MLP and LLM) is kept frozen, and only the ViR is trained.
Specifically, we first compute the loss ratio for each patch:

$$
r_i = \frac{\mathcal{L}_\text{ViCO}\big(y_i \mid I_{\frac{1}{16}}\big)}{\mathcal{L}_\text{ViCO}\big(y_i \mid I_{\frac{1}{4}}\big)},
$$

which quantifies the relative increase in loss caused by compressing the visual tokens. Based on this ratio, the binary ground-truth label for the patch router is defined as:

$$
y_i^\text{router} =
\begin{cases}
0, & r_i < \tau \; \text{(compression has negligible impact)} \\
1, & r_i \ge \tau \; \text{(compression has significant impact)},
\end{cases}
$$

where \(y_i^{\text{router}}=0\) and \(y_i^{\text{router}}=1\)  indicate that the compression rate \(\xi\) is set to \(\tfrac{1}{16}\) and \(\tfrac{1}{4}\), respectively.

> Please see [our paper](https://huggingface.co/papers/2508.18265) for more technical and experimental details.


### Test-Time Scaling


Test-time scaling (TTS) has been empirically demonstrated as an effective approach to enhance the reasoning capabilities of LLMs and MLLMs, particularly for complex tasks necessitating multi-step inference.
In this work, we implement a comprehensive test-time scaling approach that simultaneously improves reasoning depth (i.e., deep thinking) and breadth (i.e., parallel thinking).

`Deep Thinking`: By activating the Thinking mode, we guide the model to deliberately engage in step-by-step reasoning (i.e., decomposing complex problems into logical steps and validating intermediate conclusions) prior to generating the final answer. This approach systematically improves the logical structure of solutions for complex problems, particularly those requiring multi-step inference, and enhances reasoning depth.

`Parallel Thinking`: Following InternVL3, for reasoning tasks, we adopt the Best-of-N (BoN) strategy by employing [VisualPRM-v1.1](https://huggingface.co/OpenGVLab/VisualPRM-8B-v1_1) as the critic model to select the optimal response from multiple reasoning candidates.
This approach improves reasoning breadth.

> Notably, unless otherwise specified, the experimental results reported in our paper are obtained without applying TTS. Thus far, we have only applied TTS to reasoning benchmarks, since we found that the model already exhibits strong perception and understanding capabilities, and initiating TTS yields no significant improvement.


### Decoupled Vision-Language Deployment

In multimodal inference, the vision encoder and language model have distinct computational characteristics. The vision encoder that transforms images into semantic features is highly parallelizable and does not rely on long-term history state.  In contrast,  the language model adopts the inference in an autoregressive manner, which requires previous states to compute the next one. This sequential property makes the language part more sensitive to memory bandwidth and latency. 
When MLLMs are deployed online at scale, the vision and language models often block each other, thus incurring additional inference cost. This effect becomes more pronounced with larger vision models or higher-resolution images.

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/DvD.jpg)

As shown in the Figure above, we propose decoupled vision-language deployment (DvD) to address this issue by separating vision and language processing, with a particular focus on optimizing the prefilling stage. The vision subsystem batches and processes images to produce compact feature embeddings, which are then transmitted to the language subsystem for fusion with the text context prior to decoding. This separation alleviates blocking and brings multimodal prefilling performance closer to that of pure language models.
In our system implementation, the ViT and MLP (and ViR for InternVL3.5-Flash) are deployed on the vision server, while the language server executes only the LLM. The communication is unidirectional, transmitting BF16 visual features over TCP, with RDMA optionally employed to achieve higher transmission speed. Vision processing, feature transmission, and language processing are organized into an asynchronous three-stage pipeline, enabling overlapped execution and minimizing pipeline stalls.


DvD increases GPU utilization and processing efficiency on the vision side, while enabling the language server to focus exclusively on the LLM’s prefilling and decoding without being blocked by vision computation. This design leads to improved throughput and responsiveness. Moreover, the architecture supports independent hardware cost optimization for the vision and language modules, and facilitates the seamless integration of new modules without requiring modifications to the language server deployment.


## Evaluation on Multimodal Capability

### Multimodal Reasoning and Mathematics

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/performance_reasoning.jpg)

### OCR, Chart, and Document Understanding

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/performance_ocr.jpg)

### Multi-Image Understanding & Real-World Comprehension

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/performance_multi_images.jpg)

### Comprehensive Multimodal Understanding & Multimodal Hallucination Evaluation

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/performance_comprehensive.jpg)

### Visual Grounding

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/performance_grounding.jpg)

### Multimodal Multilingual Understanding

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/performance_multilingual.jpg)

### Video Understanding

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/performance_video.jpg)

### GUI Tasks

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/performance_gui.jpg)

### Embodied Tasks

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/performance_embody.jpg)

### SVG Tasks

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/performance_svg.jpg)

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/performance_svg_gen.jpg)

## Evaluation on Language Capability

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/performance_text.jpg)

## Ablation Study

### Cascade Reinforcement Learning

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/ablation_cascade_rl.jpg)

![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/ablation_cascade_rl_table.jpg)

### Decoupled Vision-Language Deployment


![image/jpg](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/ablation_dvd.jpg)

## Quick Start

We provide an example code to run `InternVL3.5-8B-HF` using `transformers`. Please note that our models with up to 30B parameters can be deployed on a single A100 GPU, while the 38B model requires two A100 GPUs and the 235B model requires eight A100 GPUs.

> In most cases, both [LMDeploy](https://github.com/InternLM/lmdeploy) and [vLLM](https://github.com/vllm-project/vllm) can be used for model deployment. However, for InternVL3.5-20B-A4B, we recommend using vLLM since lmdeploy has not yet supported GPT-OSS.

> Please use transformers>=4.52.1 to ensure the model works normally. For the 20B version of our model, transformers>=4.55.0 is required.

### Model Loading

#### 16-bit (bf16 / fp16)

```python
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
path = "OpenGVLab/InternVL3_5-8B-HF"
model = AutoModelForImageTextToText.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
```

#### BNB 8-bit Quantization

```python
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
path = "OpenGVLab/InternVL3_5-8B-HF"
model = AutoModelForImageTextToText.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
```

#### Multiple GPUs

```python
import math
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText

path = "OpenGVLab/InternVL3_5-8B-HF"
model = AutoModelForImageTextToText.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map="auto").eval()
```

### Thinking Mode

To enable thinking mode, please set the system prompt to our Thinking System Prompt. When enabling Thinking mode, we recommend setting `do_sample=True` and `temperature=0.6` to mitigate undesired repetition.

```python
R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": R1_SYSTEM_PROMPT},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "xxx"},
        ],
    },
]
```

### Inference with Transformers

The HuggingFace format checkpoints of our models are fully consistent with the APIs of the official HuggingFace models. For details, please refer to the official [documentation](https://huggingface.co/docs/transformers/v4.55.4/en/model_doc/internvl).

## Finetune

Many repositories now support fine-tuning of the InternVL series models, including [InternVL](https://github.com/OpenGVLab/InternVL), [SWIFT](https://github.com/modelscope/ms-swift), [XTuner](https://github.com/InternLM/xtuner), and others. Please refer to their documentation for more details on fine-tuning.

## Deployment

### LMDeploy

LMDeploy is a toolkit for compressing, deploying, and serving LLMs & VLMs.

```sh
pip install lmdeploy>=0.9.1
```

LMDeploy abstracts the complex inference process of multi-modal Vision-Language Models (VLM) into an easy-to-use pipeline, similar to the Large Language Model (LLM) inference pipeline.

#### A 'Hello, world' Example

```python
from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')

# Please set tp=2 for the 38B version and tp=8 for the 241B-A28B version.
model = 'OpenGVLab/InternVL3_5-8B'
pipe = pipeline(model, backend_config=PytorchEngineConfig(session_len=32768, tp=1))

response = pipe(('describe this image', image))
print(response.text)
```

#### Multi-images Inference

When dealing with multiple images, you can put them all in one list. Keep in mind that multiple images will lead to a higher number of input tokens, and as a result, the size of the context window typically needs to be increased.

```python
from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

# Please set tp=2 for the 38B version and tp=8 for the 241B-A28B version.
model = 'OpenGVLab/InternVL3_5-8B'
pipe = pipeline(model, backend_config=PytorchEngineConfig(session_len=32768, tp=1))

image_urls=[
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg',
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg'
]

images = [load_image(img_url) for img_url in image_urls]
# Numbering images improves multi-image conversations
response = pipe((f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\ndescribe these two images', images))
print(response.text)
```

#### Batch Prompts Inference

Conducting inference with batch prompts is quite straightforward; just place them within a list structure:

```python
from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image

# Please set tp=2 for the 38B version and tp=8 for the 241B-A28B version.
model = 'OpenGVLab/InternVL3_5-8B'
pipe = pipeline(model, backend_config=PytorchEngineConfig(session_len=32768, tp=1))

image_urls=[
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg",
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg"
]
prompts = [('describe this image', load_image(img_url)) for img_url in image_urls]
response = pipe(prompts)
print(response)
```

#### Multi-turn Conversation

There are two ways to do the multi-turn conversations with the pipeline. One is to construct messages according to the format of OpenAI and use above introduced method, the other is to use the `pipeline.chat` interface.

```python
from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

# Please set tp=2 for the 38B version and tp=8 for the 241B-A28B version.
model = 'OpenGVLab/InternVL3_5-8B'
pipe = pipeline(model, backend_config=PytorchEngineConfig(session_len=32768, tp=1))

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg')
gen_config = GenerationConfig(top_k=50, top_p=0.95, temperature=0.6, max_new_tokens=8192)
sess = pipe.chat(('describe this image', image), gen_config=gen_config)
print(sess.response.text)
sess = pipe.chat('What is the woman doing?', session=sess, gen_config=gen_config)
print(sess.response.text)
```

#### Service

LMDeploy's `api_server` enables models to be easily packed into services with a single command. The provided RESTful APIs are compatible with OpenAI's interfaces. Below are an example of service startup:

```shell
lmdeploy serve api_server OpenGVLab/InternVL3_5-8B --server-port 23333 --tp 1 --backend pytorch
```

To use the OpenAI-style interface, you need to install OpenAI:

```shell
pip install openai
```

Then, use the code below to make the API call:

```python
from openai import OpenAI

client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': 'describe this image',
        }, {
            'type': 'image_url',
            'image_url': {
                'url':
                'https://modelscope.oss-cn-beijing.aliyuncs.com/resource/tiger.jpeg',
            },
        }],
    }],
    temperature=0.8,
    top_p=0.8)
print(response)
```

## License

This project is released under the apache-2.0 License. This project uses the pre-trained Qwen3 as a component, which is licensed under the apache-2.0 License.

## Citation

If you find this project useful in your research, please consider citing:

```BibTeX
@article{wang2025internvl3_5,
  title={InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency},
  author={Wang, Weiyun and Gao, Zhangwei and Gu, Lixin and Pu, Hengjun and Cui, Long and Wei, Xingguang and Liu, Zhaoyang and Jing, Linglin and Ye, Shenglong and Shao, Jie and others},
  journal={arXiv preprint arXiv:2508.18265},
  year={2025}
}
```
