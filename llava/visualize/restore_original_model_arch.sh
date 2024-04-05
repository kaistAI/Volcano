#!/bin/bash

# Define file paths
repo_root=$(git rev-parse --show-toplevel)

original_file="$repo_root/llava/model/llava_arch.py"
backup_file="$repo_root/llava/model/llava_arch_backup.py"

# Restore the original files from backup
mv "$backup_file" "$original_file"

echo "Original file restored. Now llava_arch.py != llava_arch_for_image_attention.py."