# 🌋 Volcano: Mitigating Multimodal Hallucination through Self-Feedback Guided Revision
\[NAACL 2024\] This is the official github for "Volcano: Mitigating Multimodal Hallucination through Self-Feedback Guided Revision". <br><br>
Volcano employs a single LMM to generate initial responses, feedback, and revisions, as well as decisions to accept revisions. It follows a sequential procedure of an iterative critique-revision-decide loop. <br>
- [Paper](https://arxiv.org/abs/2311.07362) <br>
- Model weights ([7B](https://huggingface.co/kaist-ai/volcano-7b), [13B](https://huggingface.co/kaist-ai/volcano-13b))
- [Training dataset](https://huggingface.co/datasets/kaist-ai/volcano-train)
## News
\[Mar 14, 2023\] Our work has been accepted NAACL 2024 main conference! See you Mexico City 🇲🇽
\[Nov 14, 2023\] We released the first version of Volcano! Check out the paper, model and training dataset.
## Overview
![figure2_final](https://github.com/kaistAI/Volcano/assets/72010172/b3f2389d-c1a8-4fd7-921d-0f06de826ae0)
Large multimodal models (LMMs) suffer from multimodal hallucination, where they provide incorrect responses misaligned with the given visual information. Previous work shows that the cause of this issue is that the vision encoder fails to ground the image properly. We propose a novel approach that leverages self-feedback as visual cues, guiding the model to mitigate the hallucination in its own response. Building on this approach, we introduce **Volcano**, a multimodal self-feedback guided revision model. Volcano generates natural language feedback to its initial response based on the provided visual information and utilizes this feedback to self-revise its initial response. Volcano effectively reduces multimodal hallucination and achieves state-of-the-art on MMHal-Bench, POPE, and GAVIE. It also improves on general multimodal abilities and outperforms previous models on MM-Vet and MMBench. Through a qualitative analysis, we show that Volcano's feedback is better grounded in the image than the initial response. This means that Volcano can provide itself with richer visual information, helping alleviate multimodal hallucination. We publicly release Volcano models of 7B and 13B sizes along with the data and code.
## Setup
```bash
conda create -n volcano python=3.10 -y
conda activate volcano
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
## Input and Output Format of Volcano
```
# Critique
Generate the feedback given initial answer referring to question and image.
Question: {Question}
Initial answer: {Initial answer}

# Revise
Adjust the initial response considering the feedback and image.
Question: {Question}
Initial answer: {Initial answer}
Feedback: {Feedback}

# Decide
A. {Initial answer}
B. {Revised answer}
Answer with the option's letter from the given choices directly.
```
Volcano generates an initial response and then repeats the self-revision process a total of 3 times before producing the final answer as the determined response. You can check the [training data](https://huggingface.co/datasets/kaist-ai/volcano-train) for Volcano.
## Train
We use [LLaVA](https://github.com/haotian-liu/LLaVA) codebase in developing Volcano. Therefore, the following training & inference script is tailored to this. If you plan to start from a different VLM codebase, you should adapt the format of the data to suit your custom code.
```
deepspeed --include llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version plain \
    --data_path TRAINING_DATA_PATH \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
```
## Inference
We provide the inference code for multimodal hallucination benchmarks. (MMHal-Bench, Pope, GAVIE)
```bash
# MMHal-Bench
python -m llava.eval.volcano_mmhal_bench \
    --model_path kaist-ai/volcano-13b \
    --model_base liuhaotian/llava-v1.5-13b

# Pope
python -m llava.eval.volcano_pope \
    --model-path kaist-ai/volcano-13b \
    --model-base liuhaotian/llava-v1.5-13b \
    --image-folder <IMAGE_FOLDER> \
    --question-file <QUESTION_FILE> \
    --answers-file <ANSWER_FILE>

# GAVIE
python -m llava.eval.volcano_gavie \
    --model-path kaist-ai/volcano-13b \
    --model-base liuhaotian/llava-v1.5-13b \
    --input <INPUT> \
    --output <OUTPUT>
```
## Citation
```
@misc{lee2023volcano,
      title={Volcano: Mitigating Multimodal Hallucination through Self-Feedback Guided Revision}, 
      author={Seongyun Lee and Sue Hyun Park and Yongrae Jo and Minjoon Seo},
      year={2023},
      eprint={2311.07362},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```