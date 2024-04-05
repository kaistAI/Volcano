#!/bin/bash

# Define file paths
repo_root=$(git rev-parse --show-toplevel)

original_file="$repo_root/llava/model/llava_arch.py"
replacement_file="$repo_root/llava/model/llava_arch_for_image_attention.py"
backup_file="$repo_root/llava/model/llava_arch_backup.py"


# Backup the original file
cp "$original_file" "$backup_file"

# Replace the original file with the replacement file
cp "$replacement_file" "$original_file"

echo "Backup and replacement done. Now llava_arch.py == llava_arch_for_image_attention.py != llava_arch_backup.py."
