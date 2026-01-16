#!/bin/bash

# Default values
target_dir=""
model_pattern=""
desc="Baseline"
notes="Test"
dry_run=0

# Argument parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--dir) target_dir="$2"; shift ;;
        -m|--model) model_pattern="$2"; shift ;;
        --desc) desc="$2"; shift ;;
        --notes) notes="$2"; shift ;;
        --dry-run) dry_run=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$target_dir" ] || [ -z "$model_pattern" ]; then
    echo "Usage: sh run_all.sh --dir <directory> --model <model_pattern> [--desc <desc>] [--notes <notes>] [--dry-run]"
    exit 1
fi

echo "Searching for scripts matching '*${model_pattern}*.sh' in ${target_dir}..."

find "${target_dir}" -name "*${model_pattern}*.sh" | sort | while read script; do
    if [ $dry_run -eq 1 ]; then
        echo "[DRY RUN] Would execute: sh ${script} \"$desc\" \"$notes\""
    else
        echo "Running ${script}..."
        sh "${script}" "$desc" "$notes"
        if [ $? -ne 0 ]; then
             echo "Error running ${script}"
             exit 1
        fi
    fi
done
