import torch
import os
import fnmatch
import argparse
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import islice

LEADING_SPACE = 9601
PLOT_ONE_ROW = True
max_cutoff_quantile = 0.995
cmap = plt.get_cmap('viridis')


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := list(islice(it, n))):
        yield batch
        

def find_subdir_with_prefix(directory, prefix):
    for root, dirs, files in os.walk(directory):
        for dir in fnmatch.filter(dirs, prefix+'*'):
            return os.path.join(root, dir)
    return None


def extract_target_indices(foutput_tokens, target_tokens):
    foutput_tokens_stripped = [token[1:] if ord(token[0]) == LEADING_SPACE else token for token in foutput_tokens]
    num_target = len(target_tokens)
    for i, token in enumerate(foutput_tokens_stripped):
        start = i
        found = False
        for j in range(num_target):
            end = i + j
            if foutput_tokens_stripped[i+j] in target_tokens[j]:
                found = True
            else:
                break
        end += 1
        if found and end - start == num_target:
            return start, end
    return None, None


def create_fig_title(tag, target_tokens):
    title = f"{tag} ('"
    for i, batch in enumerate(batched(target_tokens, 3)):
        for token in batch:
            if ord(token[0]) == LEADING_SPACE:
                title += ' ' + token[1:]
            else:
                title += token
        title += '\n'
    title = title.replace("(' ", "('")[:-1] + "')"  # remove extra space at the beginning, trailing newline
    return title


def create_image_heatmap(vision_encoder_name, initial_weights_dict, feedback_weights_dict, save_path):
    items = vision_encoder_name.split("patch")[-1].split('-')  # 14 or 32
    patch_size = int(items[0])  # 14 or 32
    if len(items) == 2:
        resolution = int(items[1])  # 336
    else:
        resolution = 224
    num_patch_per_side = resolution // patch_size
    
    all_weights_dict = initial_weights_dict | feedback_weights_dict
    all_weights = torch.stack(list(all_weights_dict.values()))
    global_min = all_weights.min().item()
    global_max_cutoff = torch.quantile(all_weights, max_cutoff_quantile, interpolation="nearest").item()
    print("global_min:", global_min)
    print("global_max_cutoff:", global_max_cutoff)
    
    for key, weights in initial_weights_dict.items():
        initial_weights_dict[key] = torch.clamp(weights, max=global_max_cutoff)
    for key, weights in feedback_weights_dict.items():
        feedback_weights_dict[key] = torch.clamp(weights, max=global_max_cutoff)
        
    norm = colors.Normalize(vmin=global_min, vmax=global_max_cutoff)
    # nrows = 1 if PLOT_ONE_ROW else 2  # compared, feedback / target tokens
    # fig, axes = plt.subplots(nrows, len(target_weights_dict), figsize=(3*len(target_weights_dict), 3*nrows))
    
    if PLOT_ONE_ROW:
        nrows = 1
        fig, axes = plt.subplots(nrows, len(all_weights_dict), figsize=(3*len(all_weights_dict), 3*nrows))
        for i, (title, weights) in enumerate(all_weights_dict.items()):
            ax = axes[i]
            weights_reshaped = weights.view(num_patch_per_side, num_patch_per_side)
            sns.heatmap(weights_reshaped, ax=ax, cmap=cmap, xticklabels=False, yticklabels=False, norm=norm, cbar=False)
            ax.set_title(title)
    else:
        nrows = 2
        fig, axes = plt.subplots(nrows, len(feedback_weights_dict), figsize=(3*len(feedback_weights_dict), 3*nrows))
        for row, weights_dict in zip(range(nrows), [initial_weights_dict, feedback_weights_dict]):
            for i, (title, weights) in enumerate(weights_dict.items()):
                ax = axes[row, i]
                weights_reshaped = weights.view(num_patch_per_side, num_patch_per_side)
                sns.heatmap(weights_reshaped, ax=ax, cmap=cmap, xticklabels=False, yticklabels=False, norm=norm, cbar=False)
                ax.set_title(title)
    
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # need to adjust
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, cax=cbar_ax)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    print("Saving to", save_path)
    plt.savefig(save_path, bbox_inches='tight')
    
    save_path_pdf = save_path.replace(".png", ".pdf")
    print("Saving to", save_path_pdf)
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')


def visualize_image_attention(vision_encoder_name, qoutput_tokens, qimage_attentions, foutput_tokens, fimage_attentions, feedback_target_tokens_list, hidden_top_k, save_path):
    num_image_features = qimage_attentions[0].shape[-1]
    initial_weights = {}
    feedback_weights = {}
    qimage_attentions = qimage_attentions[:-1]  # remove last EOS token
    fimage_attentions = fimage_attentions[:-1]  # remove last EOS token
    min_output_len = min(len(qoutput_tokens), len(foutput_tokens)) - 1
    
    # compared response
    q2i_weights = torch.empty((len(qimage_attentions), num_image_features), dtype=torch.float32)
    for i, qimage_attention in enumerate(qimage_attentions):  # qoutput_len
        qimage_attention = qimage_attention.float()
        qimage_attention_fused, _ = torch.topk(qimage_attention, k=hidden_top_k, dim=0)  # (num_layers, num_head, num_image_features) -> (hidden_top_k, num_head, num_image_features)
        qimage_attention_fused = qimage_attention_fused.mean(dim=0)  # (num_head, num_image_features)
        qimage_attention_fused, _ = torch.topk(qimage_attention_fused, k=hidden_top_k, dim=0)  # (num_head, num_image_features) -> (hidden_top_k, num_image_features)
        qimage_attention_fused = qimage_attention_fused.mean(dim=0)  # (num_image_features)
        q2i_weights[i] = qimage_attention_fused
        
    if len(qoutput_tokens) - 1 > min_output_len:
        initial_all_token_weights_pooled, _ = torch.topk(q2i_weights, min_output_len, dim=0)
        initial_all_token_weights_pooled = initial_all_token_weights_pooled.mean(dim=0)
    else:
        initial_all_token_weights_pooled = q2i_weights.mean(dim=0)
    initial_weights["Initial (all tokens)"] = initial_all_token_weights_pooled  # (num_image_features)
    
    # all tokens in feedback
    f2i_weights = torch.empty((len(foutput_tokens), num_image_features), dtype=torch.float32)
    for i, fimage_attention in enumerate(fimage_attentions):  # foutput_len
        fimage_attention = fimage_attention.float()
        fimage_attention_fused, _ = torch.topk(fimage_attention, k=hidden_top_k, dim=0)
        fimage_attention_fused = fimage_attention_fused.mean(dim=0)  # (num_head, num_image_features)
        fimage_attention_fused, _ = torch.topk(fimage_attention_fused, k=hidden_top_k, dim=0)
        fimage_attention_fused = fimage_attention_fused.mean(dim=0)  # (num_image_features)
        f2i_weights[i] = fimage_attention_fused
        
    if len(foutput_tokens) - 1 > min_output_len:
        feedback_all_token_weights_pooled, _ = torch.topk(f2i_weights, min_output_len, dim=0)
        feedback_all_token_weights_pooled = feedback_all_token_weights_pooled.mean(dim=0)
    else:
        feedback_all_token_weights_pooled = f2i_weights.mean(dim=0)
        
    feedback_weights["Feedback (all tokens)"] = feedback_all_token_weights_pooled  # (num_image_features)
    
    # target tokens in feedback  
    for target_tokens in feedback_target_tokens_list:
        target_start, target_end = extract_target_indices(foutput_tokens, target_tokens)
        ftarget_tokens = foutput_tokens[target_start:target_end]
        if target_start is None:
            raise ValueError("Target tokens not found:", target_tokens)
        title = create_fig_title("Feedback", ftarget_tokens)
        
        target_token_weights_pooled = f2i_weights[target_start:target_end].mean(dim=0)
        feedback_weights[title] = target_token_weights_pooled
    
    create_image_heatmap(vision_encoder_name, initial_weights, feedback_weights, save_path)


def main(args):
    subdir = find_subdir_with_prefix(args.input_dir, str(args.instance_idx))
    if os.path.isfile(os.path.join(subdir, "output_visualization_tensor_dict.pt")):
        output_visualization_tensors = torch.load(os.path.join(subdir, "output_visualization_tensor_dict.pt"))
        qoutput_tokens = output_visualization_tensors['qoutput_tokens']
        qimage_attentions = output_visualization_tensors['qimage_attentions']
        foutput_tokens = output_visualization_tensors['best_foutput_tokens']
        fimage_attentions = output_visualization_tensors['best_fimage_attentions']
    else:
        raise ValueError("No output_visualization_tensor_dict.pt found in", subdir)
    
    tag = ''
    for target_tokens in args.feedback_target_tokens:
        tag += '_' + ','.join(target_tokens) 
    save_path = os.path.join(args.output_dir, f"image_heatmap_{tag}.png")
    visualize_image_attention(args.vision_encoder_name, qoutput_tokens, qimage_attentions, foutput_tokens, fimage_attentions, args.feedback_target_tokens, args.hidden_top_k, save_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default="./llava/visualize/artifacts", required=True)
    parser.add_argument('--output-dir', type=str, default="./llava/visualize/figures", required=True)
    parser.add_argument('--vision-encoder-name', type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--instance-idx", type=int, required=True)
    parser.add_argument("--feedback-target-tokens", type=lambda s: [str(token) for token in s.split(',')], required=True, nargs='+', help="Feedback target tokens to visualize")
    parser.add_argument("--hidden-top-k", type=int, default=3, help="top k hidden states to average for image attentions")
    args = parser.parse_args()
    
    main(args)