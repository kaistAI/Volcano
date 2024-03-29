import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import json
from tqdm import tqdm
import os
os.environ['CURL_CA_BUNDLE'] = ''

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def generate(args, qs, image_features, tokenizer, model, model_name, image_processor, context_len, conv_mode):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            # images=image_tensor,
            image_features=image_features,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


def extract_image_features(image, model, image_processor):
    images = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    if type(images) is list or images.ndim == 5:
        concat_images = torch.cat([image for image in images], dim=0)
        image_features = model.encode_images(concat_images)
        split_sizes = [image.shape[0] for image in images]
        image_features = torch.split(image_features, split_sizes, dim=0)
        image_features = [x.flatten(0, 1) for x in image_features]
    else:
        image_features = model.encode_images(images)
    return image_features


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)


    json_data = json.load(open(args.input, 'r'))
    logs = []
    iter_cnt = {
        1: 0,
        2: 0,
        3: 0
    }
    for idx, line in tqdm(enumerate(json_data)):
        image_src = line['image_src']
        image = load_image(image_src)
        question = line['question']
        qs = question
        image_features = extract_image_features(image, model, image_processor)
        outputs = generate(args, qs, image_features, tokenizer, model, model_name, image_processor, context_len, args.conv_mode)
        initial_answer = outputs
        revision = None
        print('Initial output: ', outputs)
        for i in range(3):
            fqs = "Generate the feedback given initial answer referring to question and image.\nQuestion: " + question + "\nInitial answer: " + outputs
            feedback1 = generate(args, fqs, image_features, tokenizer, model, model_name, image_processor, context_len, args.conv_mode)
            print('feedback: ', feedback1)

            rqs = "Adjust the initial response considering the feedback and image.\nQuestion: " + question + "\nInitial answer: " + outputs + "\nFeedback: " + feedback1
            revision = generate(args, rqs, image_features, tokenizer, model, model_name, image_processor, context_len, args.conv_mode)

            answer_comparison_question = question + "\nA. " + outputs + "\nB. " + revision + "\nAnswer with the option's letter from the given choices directly."
            answer_comparison = generate(args, answer_comparison_question, image_features, tokenizer, model, model_name, image_processor, context_len, args.conv_mode)
            print('answer comparison: ', answer_comparison)
            if 'b' in answer_comparison.lower():
                outputs = revision
            elif 'a' in answer_comparison.lower():
                break

        iter_cnt[i+1] += 1
        
        logs.append({
            'question': question,
            'gold answer': line['gt_answer'],
            'initial answer': initial_answer,
            'feedback': feedback1,
            'revision': revision,
        })

        line['model_answer'] = outputs
        print('---------------------------------')
        print('question: ', question)
        print('answer: ', line['gt_answer'])
        print('initial pred: ', initial_answer)
        print('pred: ', outputs)
        print('---------------------------------')

    with open(args.output, 'w') as f:
        json.dump(json_data, f, indent="\t")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--feedback_model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--revision_model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)