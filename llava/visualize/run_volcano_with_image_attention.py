import argparse
import torch
from torch.nn.functional import pad

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from io import BytesIO
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import sys
import seaborn as sns
from matplotlib.colors import Normalize
os.environ['CURL_CA_BUNDLE'] = ''


Q_PREFIX = "initial"
F_PREFIX = "feedback"
R_PREFIX = "revision"

F_ORIG_PROMPT = "Generate the feedback given initial answer referring to question and image.\nQuestion: {question}\nInitial answer: {outputs}"
R_ORIG_PROMPT = "Adjust the initial response considering the feedback and image.\nQuestion: {question}\nInitial answer: {outputs}\nFeedback: {feedback}"
D_ORIG_PROMPT = "{question}\nA. {outputs}\nB. {revision}\nAnswer with the option's letter from the given choices directly."

F_ENGINEERED_PROMPT = "Question: {question}\nInitial response: {outputs}\nCritique the initial response given question and image. Please you should NOT use the term 'reference answer'!!!!!!!!!!"
R_ENGINEERED_PROMPT = "Adjust the initial response considering the feedback and image.\nQuestion: {question}\nInitial response: {outputs}\nFeedback: {feedback}"

AVG_IMAGE_FEATURES = False
SUM_IMAGE_FEATURES = True


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        try:
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception:
            print(f"Failed to load image from {image_file}")
            image = None
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def decode_image_token(tokenizer, input_ids, image_features_len=None):
    # Replace IMAGE_TOKEN_INDEX with the string <image>
    tokens = []
    for id in input_ids:
        if id == IMAGE_TOKEN_INDEX:
            if AVG_IMAGE_FEATURES or SUM_IMAGE_FEATURES:
                tokens.append(DEFAULT_IMAGE_TOKEN)
            elif image_features_len:
                tokens.extend([f"<patch_{i+1}>" for i in range(image_features_len)])
            else:
                raise ValueError("image_features_len must be provided if not using AVG_IMAGE_FEATURES")
        else:
            tokens.extend(tokenizer.convert_ids_to_tokens([id], skip_special_tokens=False))
    
    return tokens
 

def fuse_image_attentions_for_image_features(image_attentions, pool_method, hidden_top_k, output_top_k=None):
    # image_attentions: (generated_len, num_layers, num_heads, image_features_len)
    image_attentions = image_attentions[:-1]  # skip EOS token
    if pool_method == "mean":
        image_attentions_fused = image_attentions.mean(dim=(1, 2))
    elif pool_method == "max":
        image_attentions_fused, _ = image_attentions.max(dim=1)
        image_attentions_fused, _ = image_attentions_fused.max(dim=1)
    elif pool_method == "top_k_mean":
        image_attentions = image_attentions.float()
        image_attentions_fused_top_k, _ = torch.topk(image_attentions, hidden_top_k, dim=1)  
        image_attentions_fused = image_attentions_fused_top_k.mean(dim=1) # (generated_len, num_heads, image_features_len)
        image_attentions_fused_top_k, _ = torch.topk(image_attentions_fused, hidden_top_k, dim=1)
        image_attentions_fused = image_attentions_fused_top_k.mean(dim=1) # (generated_len, image_features_len)
            
    # aggregate along generated_len dimension
    if output_top_k is not None:
        # average top-l weights across output tokens, where l is the shorter output length (l = output_top_k)
        # print("output_top_k", output_top_k)
        # print("image_attentions_fused", image_attentions_fused.shape)
        image_attentions_fused = image_attentions_fused.float()  # topk doesn't support HalfTensor
        image_attentions_fused_top_k, _ = torch.topk(image_attentions_fused, output_top_k, dim=0)
        image_attentions_fused = image_attentions_fused_top_k.mean(dim=0)
    elif pool_method == "mean":
        image_attentions_fused = image_attentions_fused.mean(dim=0)
    elif pool_method == "max":
        image_attentions_fused, _ = image_attentions_fused.max(dim=0)
    return image_attentions_fused  # (image_features_len)
    
    
def fuse_image_attentions_for_generated(image_attentions):
    image_attentions_fused = image_attentions.sum(dim=-1)  # (generated_len, num_layers, num_heads)
    image_attentions_fused = torch.linalg.norm(image_attentions_fused, dim=-1)  # (generated_len, num_layers)
    image_attentions_fused = torch.linalg.norm(image_attentions_fused, dim=-1)  # (generated_len,)
    return image_attentions_fused 
    

def visualize_avg_image_attention_all_instances(vision_encoder_name, image_attentions_list, pool_method, hidden_top_k, qoutput_tokens_len_list, save_dir, file_prefix_list):
    """
    Args:
        vision_encoder_name (str): model name of vision encoder
        image_attentions_list (list[torch.FloatTensor]): List of 2 or 3 tensors for image attentions
            Each tensor: (generated_len, num_layers, num_heads, image_features_len)
        
        pool_method: pooling method for image attentions [max, mean, top_k_mean], default is top_k_mean
        hidden_top_k: top k hidden states across layers and attention heads to average for image attentions
        qoutput_tokens_len_list (list[int]): List of output token lengths for each sample
        save_dir (str): directory to save the figure, e.g., attention/MMHal-Bench/0_{image_id}
        file_prefix_list (list[str])
    """
    
    items = vision_encoder_name.split("patch")[-1].split('-')  # 14 or 32
    patch_size = int(items[0])  # 14 or 32
    if len(items) == 2:
        resolution = int(items[1])  # 336
    else:
        resolution = 224
    num_patch_per_side = resolution // patch_size
    
    # Make sure all attentions have the correct shape
    # for image_attentions in image_attentions_list:
    #     assert image_attentions.shape[-1] == num_patch_per_side**2, \
    #         f'image_attentions.shape[-1] = {image_attentions.shape[-1]} != {num_patch_per_side**2}'
    
    image_attentions_fused_pooled_list = []
    pool_methods = [pool_method]  #["mean", "max"]
    global_min = float('inf')
    global_max = float('-inf')
    
    # Initialize tensors
    image_attentions_fused_pooled_tensor = torch.empty((len(image_attentions_list), len(image_attentions_list[0]), len(pool_methods), 576))

    if qoutput_tokens_len_list is not None:
        # truncate image attentions to the length of the output tokens
        for i, (sample_image_attentions, sample_qoutput_len) in enumerate(zip(image_attentions_list, qoutput_tokens_len_list)):
            for j, image_attentions in enumerate(sample_image_attentions):
                for k, pool_method in enumerate(pool_methods):
                    image_attentions_fused_pooled = fuse_image_attentions_for_image_features(image_attentions, pool_method=pool_method, hidden_top_k=hidden_top_k, output_top_k=sample_qoutput_len)  # (576,)
                    image_attentions_fused_pooled_tensor[i, j, k] = image_attentions_fused_pooled
    else:
        for i, sample_image_attentions in enumerate(image_attentions_list):
            for j, image_attentions in enumerate(sample_image_attentions):
                for k, pool_method in enumerate(pool_methods):
                    image_attentions_fused_pooled = fuse_image_attentions_for_image_features(image_attentions, pool_method=pool_method, hidden_top_k=hidden_top_k)  # (576,)
                    image_attentions_fused_pooled_tensor[i, j, k] = image_attentions_fused_pooled

    image_attentions_fused_pooled_tensor = torch.mean(image_attentions_fused_pooled_tensor, dim=0)  # (3, 1, 576)
    global_min = torch.min(image_attentions_fused_pooled_tensor).item()
    global_max = torch.max(image_attentions_fused_pooled_tensor).item()
    # Iterate over the first dimension (which previously was the second dimension)
    for i in range(image_attentions_fused_pooled_tensor.size(0)):
        image_attentions_pool_method_list = []
        # Iterate over the second dimension (which previously was the third dimension)
        for j in range(image_attentions_fused_pooled_tensor.size(1)):
            # The last dimension remains a tensor
            attentions_tensor = image_attentions_fused_pooled_tensor[i, j]
            image_attentions_pool_method_list.append(attentions_tensor)
        image_attentions_fused_pooled_list.append(image_attentions_pool_method_list)
        
    norm = Normalize(vmin=global_min, vmax=global_max)
    num_pool_methods = len(pool_methods)
    num_attn_maps = len(image_attentions_fused_pooled_list)
    fig, axes = plt.subplots(num_pool_methods, num_attn_maps, figsize=(4*num_attn_maps, 4*num_pool_methods))

    # single row
    if len(pool_methods) == 1:
        for col, (image_attentions_fused_pooled, file_prefix) in enumerate(zip(image_attentions_fused_pooled_list, file_prefix_list)):
            image_attentions_fused_pooled = image_attentions_fused_pooled[0].view(num_patch_per_side, num_patch_per_side)
            file_prefix = file_prefix.split('_')[0].capitalize()
            sns.heatmap(image_attentions_fused_pooled, ax=axes[col], cmap='viridis', xticklabels=False, yticklabels=False, norm=norm, cbar=False)
            axes[col].set_title(f"{file_prefix}")
    # multiple rows
    else:
        for row, pool_method in enumerate(pool_methods):
            for col, file_prefix in enumerate(file_prefix_list):
                image_attentions_fused_pooled = image_attentions_fused_pooled_list[col][row].view(num_patch_per_side, num_patch_per_side)
                file_prefix = file_prefix.split('_')[0].capitalize()
                sns.heatmap(image_attentions_fused_pooled, ax=axes[row, col], cmap='viridis', xticklabels=False, yticklabels=False, norm=norm, cbar=False)
                axes[row, col].set_title(f"{file_prefix} ({pool_method})")

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # need to adjust
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    fig.colorbar(sm, cax=cbar_ax)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    tag = file_prefix_list[-1].split('_', 1)[1]
    save_path = os.path.join(save_dir, f"{tag}_image_attention_{'_'.join(pool_methods)}.png")
    print(f"Saving image attentions to {save_path}")
    plt.savefig(save_path, bbox_inches='tight')
    
    save_path_pdf = save_path.replace('.png', '.pdf')
    print(f"Saving image attentions to {save_path_pdf}")
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')



def prepare_attention_visualization(input_ids, input_output_ids, output_attend_to_input, tokenizer):
    """Prepare attention weights and input output tokens for visualization

    Args:
        input_ids (torch.LongTensor): (batch_size*1, input_sequence_len)
        input_ids (torch.LongTensor): (batch_size*1, input_output_sequence_len)
        output_attend_to_input (tuple(tuple(torch.FloatTensor)): (generated_len (num_layers=32 (1*batch_size, num_heads=32, generated_len, input_output_sequence_len)))
        tokenizer (transformers.PreTrainedTokenizer): tokenizer object
    Returns:
        input_tokens (List[str]): input tokens starting from the image token, with the image token representing averaged/summed attentions to image features
        output_tokens (List[str]): generated output tokens
        weights_matrix (torch.FloatTensor): (num_layers, generated_len, num_heads, final_seq_len)
        autoregressive_mask (np.ndarray(bool)): (final_seq_len, generated_len). for each generated token, False for input + itself, True for future tokens
        image_attentions (torch.FloatTensor): (generated_len, num_layers, num_heads, image_features_len)
    """
    image_start_idx = torch.where(input_ids == IMAGE_TOKEN_INDEX)[-1].item()  # -1 to ignore singleton batch dimension
    image_attention_mask = torch.load(os.getenv("SAVE_IMAGE_ATTENTION_MASK_PATH")).bool()
    
    assert input_ids.shape[0] == 1, 'batch_size > 1 is not supported yet'
    assert input_output_ids.shape[0] == 1, 'batch_size > 1 is not supported yet'
    input_ids = input_ids.squeeze()
    input_output_ids  = input_output_ids.squeeze()
    image_attention_mask = image_attention_mask.squeeze()
    
    num_layers = len(output_attend_to_input[-1])
    num_heads = output_attend_to_input[0][0].shape[1]  # first generated token, first layer
    
    image_features_len = torch.sum(image_attention_mask).item()
    generated_len = len(output_attend_to_input)
    if AVG_IMAGE_FEATURES or SUM_IMAGE_FEATURES:
        final_seq_len = len(input_output_ids[image_start_idx:])
    else:
        final_seq_len = len(input_output_ids[image_start_idx:]) + image_features_len - 1  # -1 for the image token
    
    weights_matrix = torch.zeros((num_layers, generated_len, num_heads, final_seq_len), dtype=output_attend_to_input[0][0].dtype)
    image_attentions = torch.zeros((generated_len, num_layers, num_heads, image_features_len), dtype=output_attend_to_input[0][0].dtype)
    
    for generated_token_idx, generated_token_attentions in enumerate(output_attend_to_input):
        if generated_token_idx > 0:  # skip input only, only care about output attend to input
            image_attention_mask = pad(image_attention_mask, (0, 1), value=False)
        
        for layer_idx, layer_attention in enumerate(generated_token_attentions):  # (1*batch_size, num_heads, generated_length, sequence_length)
            if generated_token_idx == 0:
                layer_attention = layer_attention[:, :, -1, :]  # last index is the first output token
            layer_attention = layer_attention.squeeze()  # (num_heads, sequence_length)
            layer_non_image_attentions = layer_attention[:, ~image_attention_mask][:, image_start_idx:]  # (num_heads, sequence_length_truncated)
            layer_image_attentions = layer_attention[:, image_attention_mask]  # (num_heads, image_features_len)
            image_attentions[generated_token_idx, layer_idx] = layer_image_attentions  
            
            if AVG_IMAGE_FEATURES:
                layer_image_attentions = layer_image_attentions.mean(dim=1, keepdim=True)  # (num_heads, 1)
            elif SUM_IMAGE_FEATURES:
                layer_image_attentions = layer_image_attentions.sum(dim=1, keepdim=True)
            layer_attention_reduced = torch.cat((layer_image_attentions, layer_non_image_attentions), dim=1)
            layer_attention_reduced = pad(layer_attention_reduced, (0, final_seq_len - layer_attention_reduced.shape[1]), value=0)
            weights_matrix[layer_idx, generated_token_idx] = layer_attention_reduced
    # print("input_ids:", len(input_ids), input_ids)
    # print("input_output_ids:", len(input_output_ids), input_output_ids)
    # print("output_attend_to_input:", len(output_attend_to_input))
    # for d in output_attend_to_input:
        # print(len(d), d[0].shape)
    # print("output_tokens:", len(output_tokens))
    # print("generated_len:", generated_len)
    weights_matrix = weights_matrix.detach().cpu()
    image_attentions = image_attentions.detach().cpu()
    
    autoregressive_mask = torch.full((final_seq_len, generated_len), False, dtype=torch.bool) # True for data not to visualize (forward attention)
    for generated_idx in range(generated_len):
        pad_cnt = generated_len - generated_idx - 1
        if pad_cnt == 0:
            break
        autoregressive_mask[-pad_cnt:, generated_idx] = True
    autoregressive_mask = autoregressive_mask.numpy()
    
    input_tokens = decode_image_token(tokenizer, input_ids[image_start_idx:], image_features_len)
    output_tokens = tokenizer.convert_ids_to_tokens(input_output_ids[len(input_ids):])
    
    assert len(output_tokens) == generated_len
    return input_tokens, output_tokens, weights_matrix, autoregressive_mask, image_attentions
    

def generate_for_attention(args, qs, image_features, tokenizer, model, model_name, conv_mode):
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
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()  # text
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    input_token_len = input_ids.shape[1]
    with torch.inference_mode():
        input_output = model.generate(  # SampleDecoderOnlyOutput or GreedyDecoderOnlyOutput
            input_ids,
            image_features=image_features,
            do_sample=True if args.sample else False,
            temperature=0.5,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            output_attentions=True,
            return_dict_in_generate=True
        )
    input_output_ids = input_output.sequences
    input_tokens, output_tokens, weights_matrix, autoregressive_mask, image_attentions = prepare_attention_visualization(
        input_ids=input_ids, 
        input_output_ids=input_output_ids, 
        output_attend_to_input=input_output.attentions, 
        tokenizer=tokenizer)
        
    n_diff_input_output = (input_ids != input_output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(input_output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    # print(outputs)
    
    return outputs, input_tokens, output_tokens, weights_matrix, autoregressive_mask, image_attentions


def generate(args, qs, image_features, tokenizer, model, model_name, conv_mode):
    # Only used for answer comparison
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

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            image_features=image_features,
            do_sample=False,  # greedy decoding
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


def create_file_prefixes(args):
    qfile_prefix = Q_PREFIX
    ffile_prefix = F_PREFIX
    rfile_prefix = R_PREFIX
    
    # included in image attention visualization file name
    if args.sample:
        qfile_prefix += "_sample"
        ffile_prefix += "_sample"
        rfile_prefix += "_sample"
    else:
        qfile_prefix += "_greedy"
        ffile_prefix += "_greedy"
        rfile_prefix += "_greedy"
    
    if args.engineered_prompt:
        ffile_prefix += "_engineered"
        rfile_prefix += "_engineered"
    
    file_prefixes = [qfile_prefix, ffile_prefix, rfile_prefix]
        
    return qfile_prefix, ffile_prefix, rfile_prefix, file_prefixes

        
def run_volcano_with_attention(args):
    disable_torch_init()
    
    print(f"Loading model from {args.model_path}")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    vision_encoder_name = model.get_vision_tower().vision_tower_name
    
    print(f"Loading input data from {args.input}")
    json_data = json.load(open(args.input, 'r'))
    logs = []
    
    if args.engineered_prompt:
        fqs_template = F_ENGINEERED_PROMPT
        rqs_template = R_ENGINEERED_PROMPT
    else:
        fqs_template = F_ORIG_PROMPT
        rqs_template = R_ORIG_PROMPT
        
    min_output_tokens_len_list = []
    image_attentions_total_list = []
    for idx, line in tqdm(enumerate(json_data)):
        if args.instance_idx is not None and idx not in args.instance_idx:
            continue
        image_id = line['image_id']
        image_src = line['image_src']
        image = load_image(image_src)
        if image is None:
            continue
        question = line['question']
        gold = line['gold_answer']
        
        image_features = extract_image_features(image, model, image_processor)
        
        instance_output_dir = os.path.join(args.output_dir, f"{idx}_{image_id}")
        os.makedirs(instance_output_dir, exist_ok=True)
        
        qs = question
        outputs, qinput_tokens, qoutput_tokens, qweights_matrix, qautoregressive_mask, qimage_attentions = generate_for_attention(args, qs, image_features, tokenizer, model, model_name, args.conv_mode)
        best_feedback, best_finput_tokens, best_foutput_tokens, best_fweights_matrix, best_fautoregressive_mask, best_fimage_attentions = None, None, None, None, None, None
        best_revision, best_rinput_tokens, best_routput_tokens, best_rweights_matrix, best_rautoregressive_mask, best_rimage_attentions = None, None, None, None, None, None
        outputs_list = [outputs]
        feedback_list = []
        revision_list = []
        for i in range(3):
            fqs = fqs_template.format(question=question, outputs=outputs)
            feedback, finput_tokens, foutput_tokens, fweights_matrix, fautoregressive_mask, fimage_attentions = generate_for_attention(args, fqs, image_features, tokenizer, model, model_name, args.conv_mode)
            rqs = rqs_template.format(question=question, outputs=outputs, feedback=feedback)
            revision, rinput_tokens, routput_tokens, rweights_matrix, rautoregressive_mask, rimage_attentions = generate_for_attention(args, rqs, image_features, tokenizer, model, model_name, args.conv_mode)
            answer_comparison_question = D_ORIG_PROMPT.format(question=question, outputs=outputs, revision=revision)
            answer_comparison = generate(args, answer_comparison_question, image_features, tokenizer, model, model_name, args.conv_mode)
            
            feedback_list.append(feedback)
            revision_list.append(revision)
            if 'b' in answer_comparison.lower():
                outputs = revision
                best_feedback, best_finput_tokens, best_foutput_tokens, best_fweights_matrix, best_fautoregressive_mask, best_fimage_attentions = feedback, finput_tokens, foutput_tokens, fweights_matrix, fautoregressive_mask, fimage_attentions
                best_revision, best_rinput_tokens, best_routput_tokens, best_rweights_matrix, best_rautoregressive_mask, best_rimage_attentions = revision, rinput_tokens, routput_tokens, rweights_matrix, rautoregressive_mask, rimage_attentions
                outputs_list.append(outputs)
            elif 'a' in answer_comparison.lower():
                break
            
        qfile_prefix, ffile_prefix, rfile_prefix, file_prefixes = create_file_prefixes(args)
        
        if args.output_visualization_tensors:
            output_visualization_tensors = {
                'qinput_tokens': qinput_tokens,
                'qoutput_tokens': qoutput_tokens,
                'qweights_matrix': qweights_matrix,
                'qautoregressive_mask': qautoregressive_mask,
                'qimage_attentions': qimage_attentions,
                'best_finput_tokens': best_finput_tokens,
                'best_foutput_tokens': best_foutput_tokens,
                'best_fweights_matrix': best_fweights_matrix,
                "best_fautoregressive_mask": best_fautoregressive_mask, 
                'best_fimage_attentions': best_fimage_attentions,
                'best_rinput_tokens': best_rinput_tokens,
                'best_routput_tokens': best_routput_tokens,
                'best_rweights_matrix': best_rweights_matrix,
                'best_rautoregressive_mask': best_rautoregressive_mask,
                'best_rimage_attentions': best_rimage_attentions
            }
            torch.save(output_visualization_tensors, os.path.join(instance_output_dir, "output_visualization_tensor_dict.pt"))
                
        if args.visualize_avg_attention_all:
            image_attentions_list = [qimage_attentions, best_fimage_attentions, best_rimage_attentions]
            
            if best_feedback and best_revision:
                image_attentions_total_list.append(image_attentions_list[:2])  # no need to visualize revision image attentions
                min_output_token_len = min(len(qoutput_tokens), len(best_foutput_tokens)) - 1  # skip EOS token
                min_output_tokens_len_list.append(min_output_token_len)
        
        if args.log:
            logs.append({
                'image_id': image_id,
                'image_src': image_src,
                'question': question,
                'gold_answer': gold,
                'outputs': outputs_list,
                'feedback': feedback_list,
                'revision': revision_list,
                'stop_iter': i + 1,
            })
            print(logs[-1])
    
    if args.visualize_avg_attention_all:  # when revision is chosen
        print("Visualizing average image attentions across all instances")
        print("Number of samples where revision is chosen: ", len(image_attentions_total_list))
        # tag = file_prefixes[-1].split('_', 1)[1]
        # filename = f"image_attentions_total_list_{tag}.pt"
        # torch.save(image_attentions_total_list, os.path.join(args.output_dir, filename))
        
        # torch.save(min_output_tokens_len_list, os.path.join(args.output_dir, f"min_output_tokens_len_list_{tag}.pt"))
        visualize_avg_image_attention_all_instances(
            vision_encoder_name, 
            image_attentions_total_list, 
            args.pool_method,
            args.hidden_top_k, 
            min_output_tokens_len_list, 
            args.output_dir, 
            file_prefixes
        )
        

    if args.log:
        print(f"Saving log file to {args.log}")
        with open(args.log, 'w') as f:
            json.dump(logs, f, indent=4)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument("--model-path", type=str, default="kaist-ai/volcano-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    
    # data parameters
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--log", type=str, default="./logs.json", help="log file (.json) path")
    parser.add_argument("--instance-idx", type=lambda s: set(int(idx) for idx in s.split(',')), default=None, help="comma separated list of instance indices to visualize")
   
    # generation parameters
    parser.add_argument("--sample", action="store_true", help="use sampling instead of greedy decoding")
    parser.add_argument("--engineered-prompt", action="store_true", help="use engineered prompt instead of original prompt")
    
    # visualization mode parameters
    parser.add_argument("--visualize-avg-attention-all", action="store_true", help="visualize attention weights to images for initial and feedback, averaged across all instances")
    parser.add_argument("--output-visualization-tensors", action="store_true", help="output dict of raw attention tensors for in-depth analysis")
    # attention aggregation hyperparameters
    parser.add_argument("--pool-method", type=str, choices=["mean", "max", "top_k_mean"], default="top_k_mean", help="pooling method for image attentions")
    parser.add_argument("--hidden-top-k", type=int, default=3, help="top k hidden states to average for image attentions")
    args = parser.parse_args()

    run_volcano_with_attention(args)
