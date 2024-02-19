#!/bin/bash

PYTHON_SCRIPT_PATH = "scored.py"
INPUT_IMAGE_PATH = "test-images/b-27.jpg"
OUTPUT_IMAGE_PATH = "scored.jpg"

python3 "$PYTHON_SCRIPT_PATH" "$INPUT_IMAGE_PATH" "$OUTPUT_IMAGE_PATH"
