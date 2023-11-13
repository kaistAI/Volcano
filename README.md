# 🌋 Volcano: Mitigating Multimodal Hallucination through Self-Feedback Guided Revision
This is the official github for "Volcano: Mitigating Multimodal Hallucination through Self-Feedback Guided Revision". <br><br>
Volcano employs a single LMM to generate initial responses, feedback, and revisions, as well as decisions to accept revisions. It follows a sequential procedure of an iterative critique-revision-decide loop. <br>

## News
\[Nov 13, 2023\] We released the first version of Volcano! Check out the [paper](), model ([7b](https://huggingface.co/kaist-ai/volcano-7b), [13b](https://huggingface.co/kaist-ai/volcano-13b)) and [training dataset](https://huggingface.co/datasets/kaist-ai/volcano-train)!
## Overview
![figure2_final](https://github.com/kaistAI/Volcano/assets/72010172/267b2ba6-3895-4e46-9be3-e8a0bee984eb)

Large multimodal models (LMMs) suffer from multimodal hallucination, where they provide incorrect responses misaligned with the given visual information. Previous work shows that the cause of this issue is that the vision encoder fails to ground the image properly. We propose a novel approach that leverages self-feedback as visual cues, guiding the model to mitigate the hallucination in its own response. Building on this approach, we introduce **Volcano**, a multimodal self-feedback guided revision model. Volcano generates natural language feedback to its initial response based on the provided visual information and utilizes this feedback to self-revise its initial response. Volcano effectively reduces multimodal hallucination and achieves state-of-the-art on MMHal-Bench, POPE, and GAVIE. It also improves on general multimodal abilities and outperforms previous models on MM-Vet and MMBench. Through a qualitative analysis, we show that Volcano's feedback is better grounded in the image than the initial response. This means that Volcano can provide itself with richer visual information, helping alleviate multimodal hallucination. We publicly release Volcano models of 7B and 13B sizes along with the data and code.
