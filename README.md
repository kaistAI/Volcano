# 🌋 Volcano: Mitigating Multimodal Hallucination through Self-Feedback Guided Revision
This is the official GitHub for "Volcano: Mitigating Multimodal Hallucination through Self-Feedback Guided Revision". <br><br>
Volcano employs a single LMM to generate initial responses, feedback, and revisions, as well as decisions to accept revisions. It follows a sequential procedure of an iterative critique-revision-decide loop. <br>
- [Paper](https://arxiv.org/abs/2311.07362) <br>
- Model weights ([7B](https://huggingface.co/kaist-ai/volcano-7b), [13B](https://huggingface.co/kaist-ai/volcano-13b))
- [Training dataset](https://huggingface.co/datasets/kaist-ai/volcano-train)
## News
\[Nov 14, 2023\] We released the first version of Volcano! Check out the paper, model and training dataset.
## Overview
![figure2_final](https://github.com/kaistAI/Volcano/assets/72010172/b3f2389d-c1a8-4fd7-921d-0f06de826ae0)
Large multimodal models (LMMs) suffer from multimodal hallucination, where they provide incorrect responses misaligned with the given visual information. Previous work shows that the cause of this issue is that the vision encoder fails to ground the image properly. We propose a novel approach that leverages self-feedback as visual cues, guiding the model to mitigate the hallucination in its own response. Building on this approach, we introduce **Volcano**, a multimodal self-feedback guided revision model. Volcano generates natural language feedback to its initial response based on the provided visual information and utilizes this feedback to self-revise its initial response. Volcano effectively reduces multimodal hallucination and achieves state-of-the-art on MMHal-Bench, POPE, and GAVIE. It also improves on general multimodal abilities and outperforms previous models on MM-Vet and MMBench. Through a qualitative analysis, we show that Volcano's feedback is better grounded in the image than the initial response. This means that Volcano can provide itself with richer visual information, helping alleviate multimodal hallucination. We publicly release Volcano models of 7B and 13B sizes along with the data and code.
