import torch
import os
import fnmatch
import argparse
import matplotlib.colors as colors
import matplotlib.pyplot as plt

LEADING_SPACE = 9601
cmap = plt.get_cmap('BuGn')
alpha = 0.5
brightness_threshold = 0.6  # Set your brightness threshold here


def find_subdir_with_prefix(directory, prefix):
    for root, dirs, files in os.walk(directory):
        for dir in fnmatch.filter(dirs, prefix+'*'):
            return os.path.join(root, dir)
    return None


def create_text_heatmap(tokens, weights, save_path):
    tokens = [' ' + token[1:] if ord(token[0]) == LEADING_SPACE else token for token in tokens]  # Add a space if the token starts with a leading space
    off = (sum(weights) / len(weights)) * alpha
    normer = colors.Normalize(vmin=min(weights)-off, vmax=max(weights)+off)
    bg_colors = [cmap(normer(x)) for x in weights]  # Get RGBA colors
    hex_colors = [colors.to_hex(color) for color in bg_colors]  # Convert RGBA to hex

    if len(tokens) != len(hex_colors):
        raise ValueError("number of tokens and colors don't match")

    style_elems = []
    span_elems= []

    for i, token in enumerate(tokens):
        # Check if there is a leading space
        if ord(token[0]) == LEADING_SPACE:
            space = ' '  # A space will be added before this token
            token = token[1:]  # Remove the leading space from the token itself
        else:
            space = ''
        
        # Convert RGB to HSV, then set text color based on brightness
        hsv = colors.rgb_to_hsv(bg_colors[i][:3])  # Exclude alpha from RGBA
        text_color = '#ffffff' if hsv[2] < brightness_threshold else '#000000'
        
        # Append the styles, consider the leading space for background color
        style_elems.append(f'.c{i} {{ background-color: {hex_colors[i]}; color: {text_color}; }}')
        span_elems.append(f'{space}<span class="c{i}">{space}{token}</span>')

    # Write to the file, ensure ::before is used only when there is a leading space
    print("Saving to", save_path)
    with open(save_path, 'w') as f:
        f.write(
            f"""<!DOCTYPE html><html><head><link href="https://fonts.googleapis.com/css?family=Roboto+Mono&display=swap" rel="stylesheet"><style>span {{ font-family: "Helvetica Neue", monospace; font-size: 20px; }} {' '.join(style_elems)} {' '.join([f'.c{i}:not(:first-child)::before {{ content: " "; }}' for i in range(1, len(tokens))])}</style></head><body>{''.join(span_elems)}</body></html>"""
        )


def visualize_text_attention(output_tokens, image_attentions, hidden_top_k, save_path):
    weights = []
    
    for image_attention in image_attentions:  # generated len
        image_attention = image_attention.float()
        image_attention_fused, _ = torch.topk(image_attention, k=hidden_top_k, dim=0)  # (hidden_top_k, num_heads, num_image_features)
        image_attention_fused = torch.mean(image_attention_fused, dim=0)  # (num_heads, num_image_features)
        image_attention_fused, _ = torch.topk(image_attention_fused, k=hidden_top_k, dim=0)  # (hidden_top_k, num_image_features)
        image_attention_fused = torch.mean(image_attention_fused, dim=0)  # (num_image_features)
        image_attention_fused = torch.mean(image_attention_fused, dim=0)  # (1)
        weights.append(image_attention_fused.item())
        
    create_text_heatmap(output_tokens, weights, save_path)


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
    
    visualize_text_attention(qoutput_tokens, qimage_attentions, args.hidden_top_k, os.path.join(args.output_dir, f"text_heatmap_{args.instance_idx}_q2i_attention.html"))
    visualize_text_attention(foutput_tokens, fimage_attentions, args.hidden_top_k, os.path.join(args.output_dir, f"text_heatmap_{args.instance_idx}_f2i_attention.html"))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default="./llava/visualize/artifacts", required=True)
    parser.add_argument('--output-dir', type=str, default="./llava/visualize/figures", required=True)
    parser.add_argument("--instance-idx", type=int, required=True)
    parser.add_argument("--hidden-top-k", type=int, default=3, help="top k hidden states to average for image attentions")
    args = parser.parse_args()
    
    main(args)