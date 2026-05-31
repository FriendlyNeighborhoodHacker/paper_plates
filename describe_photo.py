#!/usr/bin/env python3
"""
describe_photo.py — Generates drawing-instructions descriptions from student photos.

Sends photos to OpenAI's vision API (gpt-4o) and saves text descriptions
to .txt files with the same name as each photo in the same directory.
The descriptions can then be used as the "drawing_instructions" column in the
Google Sheet for paper plate image generation.

Usage:
    python describe_photo.py                      # Process all photos in photos/ dir
    python describe_photo.py photos/alex.jpg      # Process a single photo

Output:
    photos/alex_weinrich.txt  (skipped if already exists; delete to regenerate)
"""

import base64
import os
import sys
import time

from openai import OpenAI

from generate_images import CONFIG_PATH, load_config

PHOTOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photos")

VISION_MODEL = "gpt-4o"

PROMPT = """I am creating a hand-drawn cartoon caricature of a high school student for a school art project. \
Please provide a detailed artist's reference description of the person in this photo. \
Do NOT identify or name the person — describe only their visible physical features.

Cover each of the following in specific detail:

FACE SHAPE: Describe the overall face shape (oval, round, square, heart, oblong, diamond). \
Describe the jawline (sharp/soft/square/tapered/rounded), chin shape (pointed/rounded/wide/cleft), \
and cheekbone prominence (high/flat/prominent).

HAIR: Exact color with nuance (e.g. "dark chestnut brown", "sandy dirty-blonde", not just "brown"). \
Texture (fine/medium/thick; straight/wavy/curly/coily). Length. How it is parted or styled. \
Volume and density. Any distinctive qualities (tousled, sleek, wispy, frizzy, layered).

EYES: Size relative to the face (large/medium/small). Shape (almond/round/hooded/deep-set/wide-set/close-set). \
Color with detail (e.g. "warm hazel with green flecks", not just "brown"). \
Lash density. Whether they appear bright/sleepy/intense/expressive.

EYEBROWS: Thickness (pencil-thin/medium/thick/bushy). Shape (strongly arched/gently arched/straight/slightly curved). \
Color. Groomed vs natural. Distance above the eyes.

NOSE: Bridge width (narrow/medium/wide) and height (flat/medium/prominent). \
Tip shape (rounded/pointed/wide/bulbous/button/upturned). Nostril shape (flared/narrow/average). \
Overall nose size relative to the face (small/medium/large/prominent).

MOUTH & LIPS: Upper lip shape and fullness (thin/medium/full; pronounced cupid's bow or flat). \
Lower lip fullness. Width of mouth (wide smile, narrow mouth, average). Any notable features.

SKIN: Specific tone (e.g. fair/light/medium/olive/tan/warm brown/deep) and undertone (warm/cool/neutral). \
Freckles (none/light scattered/moderate/heavy), moles, dimples, or other notable texture.

MOST DISTINCTIVE FEATURES: List the 2-3 features that most define this person's look — \
what a caricaturist would exaggerate to make them immediately recognizable.

CLOTHING (visible): Briefly describe what they are wearing."""

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def describe_photo(photo_path: str, client: OpenAI) -> str:
    """Send the photo to GPT-4o vision and return the text description."""
    with open(photo_path, "rb") as f:
        image_bytes = f.read()

    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    ext = os.path.splitext(photo_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(ext, "image/jpeg")

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                ],
            }
        ],
        max_tokens=2048,
    )

    return response.choices[0].message.content.strip()


def collect_pending_photos(directory: str) -> list[tuple[str, str]]:
    """
    Scan `directory` for image files that don't yet have a matching .txt file.
    Returns a list of (photo_path, output_txt_path) tuples.
    """
    if not os.path.isdir(directory):
        sys.exit(f"ERROR: Photos directory not found: '{directory}'")

    pending = []
    for filename in sorted(os.listdir(directory)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        photo_path = os.path.join(directory, filename)
        base = os.path.splitext(photo_path)[0]
        output_path = base + ".txt"
        if not os.path.exists(output_path):
            pending.append((photo_path, output_path))
    return pending


def process_one(photo_path: str, output_path: str, client: OpenAI) -> bool:
    """Generate a description for one photo and save it. Returns True on success."""
    try:
        description = describe_photo(photo_path, client)
    except Exception as e:
        print(f"\n  ❌  API call failed: {e}")
        return False

    with open(output_path, "w") as f:
        f.write(description)

    print(f"  ✅  Saved: {os.path.basename(output_path)}")
    print()
    print(description)
    print()
    return True


def main():
    print("=" * 60)
    print("  Paper Plate Awards — Photo Describer")
    print("=" * 60)

    # ── Single-file mode ──────────────────────────────────────────
    if len(sys.argv) == 2:
        photo_path = sys.argv[1]

        if not os.path.exists(photo_path):
            sys.exit(f"ERROR: File not found: '{photo_path}'")

        ext = os.path.splitext(photo_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            sys.exit(
                f"ERROR: Unsupported file type '{ext}'.\n"
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        base = os.path.splitext(photo_path)[0]
        output_path = base + ".txt"

        print(f"\n  Photo:  {photo_path}")
        print(f"  Output: {output_path}")

        if os.path.exists(output_path):
            print(f"\n  ⏭️  Already exists — skipping. (Delete to regenerate.)")
            print("=" * 60)
            return

        config = load_config(CONFIG_PATH)
        client = OpenAI(api_key=config["OPENAI_API_KEY"])
        print(f"\n  Sending to {VISION_MODEL}…\n")
        process_one(photo_path, output_path, client)
        print("=" * 60)
        return

    # ── Batch mode (no arguments): scan entire photos/ directory ──
    if len(sys.argv) != 1:
        sys.exit(
            "Usage:\n"
            "  python describe_photo.py                   # batch: all photos in photos/\n"
            "  python describe_photo.py photos/name.jpg   # single photo"
        )

    pending = collect_pending_photos(PHOTOS_DIR)

    if not pending:
        print(f"\n  ✅  All photos in '{PHOTOS_DIR}' already have descriptions.")
        print("=" * 60)
        return

    # Print the plan
    print(f"\n  Found {len(pending)} photo(s) without a description:\n")
    for i, (photo_path, output_path) in enumerate(pending, start=1):
        print(f"  [{i}] {os.path.basename(photo_path)}  →  {os.path.basename(output_path)}")

    print()
    confirm = input("  Proceed? [y/N]: ").strip().lower()
    if confirm not in ("y", "yes"):
        print("\n  Aborted.")
        print("=" * 60)
        return

    # Load config once for the whole batch
    config = load_config(CONFIG_PATH)
    client = OpenAI(api_key=config["OPENAI_API_KEY"])

    print()
    successes = 0
    failures = 0

    for i, (photo_path, output_path) in enumerate(pending, start=1):
        print(f"[{i}/{len(pending)}] {os.path.basename(photo_path)}")
        print(f"  Sending to {VISION_MODEL}…")
        ok = process_one(photo_path, output_path, client)
        if ok:
            successes += 1
        else:
            failures += 1
        # Brief pause between calls
        if i < len(pending):
            time.sleep(1)

    print("=" * 60)
    print(f"  Done!  ✅ {successes} generated  |  ❌ {failures} failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
