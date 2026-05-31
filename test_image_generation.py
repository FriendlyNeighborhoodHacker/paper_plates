#!/usr/bin/env python3
"""
test_image_generation.py — Tests the OpenAI image generation API end-to-end.

Generates a single test image of a dragon using the configured model and
saves it to generated_images/test_dragon.png. No spreadsheet data is used.

Usage:
    python test_image_generation.py
"""

import os

from generate_images import (
    CONFIG_PATH,
    DEFAULT_IMAGE_MODEL,
    IMAGE_QUALITY,
    IMAGE_SIZE,
    OUTPUT_DIR,
    generate_and_save_image,
    load_config,
)
from openai import OpenAI

TEST_PROMPT = "A friendly cartoon dragon sitting on a cloud, bold black outlines, vibrant colors, whimsical hand-drawn style."
TEST_FILENAME = "test_dragon.png"


def main():
    print("=" * 60)
    print("  Paper Plate Awards — Image Generation Test")
    print("=" * 60)

    # Load config
    print("\nLoading config…")
    config = load_config(CONFIG_PATH)
    image_model = config.get("IMAGE_MODEL", DEFAULT_IMAGE_MODEL)
    print(f"  Model: {image_model}  |  Size: {IMAGE_SIZE}  |  Quality: {IMAGE_QUALITY}")

    # Set up output directory and client
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = OpenAI(api_key=config["OPENAI_API_KEY"])

    output_path = os.path.join(OUTPUT_DIR, TEST_FILENAME)

    # Warn if already exists
    if os.path.exists(output_path):
        print(f"\n  ⚠️  '{TEST_FILENAME}' already exists — it will be overwritten.")
        os.remove(output_path)

    # Generate
    print(f"\nGenerating test image…")
    print(f"  Prompt: \"{TEST_PROMPT}\"")
    print(f"  Output: {output_path}")
    print()

    success = generate_and_save_image(
        client=client,
        prompt=TEST_PROMPT,
        output_path=output_path,
        name="test_dragon",
        image_model=image_model,
    )

    print("\n" + "=" * 60)
    if success:
        print(f"  ✅  Test passed! Image saved to:")
        print(f"      {output_path}")
    else:
        print("  ❌  Test failed — check the error above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
