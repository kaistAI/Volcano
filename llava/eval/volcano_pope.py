import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import random

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

def generate(args, input_ids, image_features, stop_str, model, tokenizer):
    with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                image_features=image_features,
                # images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True)

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

def generate2(args, question, image_features, stop_str, model, tokenizer, image_processor, model_config):
    qs = question

    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # image_tensor = process_images([image], image_processor, model_config)[0].unsqueeze(0).cuda()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                image_features=image_features,
                # images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True)

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

def generate_fr(args, qs, image_features, tokenizer, model, model_name, image_processor, context_len, conv_mode, model_config):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # image_tensor = process_images([image], image_processor, model_config)[0]
    # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            image_features=image_features,
            # images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=128,
            use_cache=True)

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
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    temp_results = []

    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        image_path = args.image_folder + '/' + line["image"]
        cur_prompt = line["text"]
        
        image = Image.open(image_path).convert('RGB')
        image_features = extract_image_features(image, model, image_processor)

        stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        outputs = generate(args, input_ids, image_features, stop_str, model, tokenizer)
        initial_answer = outputs
        for i in range(3):
            fqs = "Generate the feedback given initial answer referring to question and image.\nQuestion: " + cur_prompt + "\nInitial answer: " + outputs
            feedback1 = generate_fr(args, fqs, image_features, tokenizer, model, model_name, image_processor, context_len, args.conv_mode, model.config)
            print('feedback: ', feedback1)

            rqs = "Adjust the initial response considering the feedback and image.\nQuestion: " + cur_prompt + "\nInitial answer: " + outputs + "\nFeedback: " + feedback1
            revision = generate_fr(args, rqs, image_features, tokenizer, model, model_name, image_processor, context_len, args.conv_mode, model.config)

            revised_cur_prompt = cur_prompt.split("\n")[0] + "\n" + "Answer with the option's letter from the given choices directly."
            print('question: ', revised_cur_prompt)
            randn = random.randint(0, 1)
            if randn == 0:
                answer_comparison_question = revised_cur_prompt + "\nA. " + outputs + "\nB. " + revision
                answer_comparison = generate2(args, answer_comparison_question, image_features, stop_str, model, tokenizer, image_processor, model.config)
                if 'no' in answer_comparison.lower():
                    if outputs == 'No':
                        break
                    outputs = 'No'
                elif 'yes' in answer_comparison.lower():
                    if outputs == 'Yes':
                        break
                    outputs = 'Yes'
                elif 'a' in answer_comparison.lower():
                    outputs = outputs
                    break
                elif 'b' in answer_comparison.lower():
                    outputs = revision
            else:
                answer_comparison_question = revised_cur_prompt + "\nA. " + revision + "\nB. " + outputs
                answer_comparison = generate2(args, answer_comparison_question, image_features, stop_str, model, tokenizer, image_processor, model.config)
                if 'no' in answer_comparison.lower():
                    if outputs == 'No':
                        break
                    outputs = 'No'
                elif 'yes' in answer_comparison.lower():
                    if outputs == 'Yes':
                        break
                    outputs = 'Yes'
                elif 'a' in answer_comparison.lower():
                    outputs = revision
                elif 'b' in answer_comparison.lower():
                    outputs = outputs
                    break


        print('---------------------------------')
        print('question: ', cur_prompt)
        print('initial pred: ', initial_answer)
        print('pred: ', outputs)
        print('---------------------------------')

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        temp_results.append({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}})
        if len(temp_results) % 1000 == 0:
            with open(args.answers_file + '.temp', 'w') as f:
                for temp_result in temp_results:
                    f.write(json.dumps(temp_result) + '\n')
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)