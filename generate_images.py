#!/usr/bin/env python3
"""
Paper Plate Awards - Image Generator
Fetches student names and superlative descriptions from Google Sheets,
then generates paper plate award images using OpenAI's image generation API.
"""

import base64
import csv
import io
import os
import re
import sys
import threading
import time

import requests
import yaml
from openai import OpenAI

# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")
PROMPT_INTRO_PATH = os.path.join(SCRIPT_DIR, "prompt_intro.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "generated_images")
PHOTOS_DIR = os.path.join(SCRIPT_DIR, "photos")

IMAGE_SIZE = "1024x1024"      # Square — ideal for a circular plate composition
IMAGE_QUALITY = "high"        # Best quality available on the plan

# Seconds to wait between API calls to be polite to rate limits
INTER_REQUEST_DELAY = 3

# Default model if not specified in config
DEFAULT_IMAGE_MODEL = "gpt-image-1"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load YAML config and validate required keys."""
    if not os.path.exists(path):
        sys.exit(
            f"ERROR: Config file not found at '{path}'.\n"
            "Please copy config.yaml and fill in your credentials."
        )
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    for key in ("OPENAI_API_KEY", "GOOGLE_SHEETS_URL"):
        if not config.get(key) or config[key].startswith("your-"):
            sys.exit(
                f"ERROR: '{key}' is not set in config.yaml.\n"
                "Please fill in your credentials before running."
            )
    return config


def load_prompt_intro(path: str) -> str:
    """Read the prompt intro text file."""
    if not os.path.exists(path):
        sys.exit(f"ERROR: prompt_intro.txt not found at '{path}'.")
    with open(path, "r") as f:
        return f.read().strip()


def normalize_name_for_file(name: str) -> str:
    """
    Normalize a name to match the filename format in the photos directory.
    - Strip leading/trailing whitespace
    - Remove "Mr." and "Ms." prefixes (case-insensitive)
    - Convert to lowercase
    - Replace spaces with underscores
    """
    normalized = name.strip()
    # Remove Mr. or Ms. prefix (with optional space after the period)
    normalized = re.sub(r'^(Mr\.|Ms\.)\s*', '', normalized, flags=re.IGNORECASE)
    normalized = normalized.strip()
    # Convert to lowercase and replace spaces with underscores
    normalized = normalized.lower().replace(' ', '_')
    return normalized


def load_drawing_instructions_from_file(name: str) -> str | None:
    """
    Attempt to load drawing instructions from a text file in the photos directory.
    Returns the file contents if found, None otherwise.
    """
    normalized_name = normalize_name_for_file(name)
    file_path = os.path.join(PHOTOS_DIR, f"{normalized_name}.txt")
    
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception as e:
            print(f"  WARNING: Could not read {file_path}: {e}")
    
    return None


def sheet_url_to_csv_url(url: str) -> str:
    """
    Convert a Google Sheets edit/view URL to a CSV export URL.
    Handles URLs like:
      https://docs.google.com/spreadsheets/d/SHEET_ID/edit#gid=0
      https://docs.google.com/spreadsheets/d/SHEET_ID/edit
      https://docs.google.com/spreadsheets/d/SHEET_ID/
    """
    # Extract the sheet ID
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        sys.exit(
            "ERROR: Could not extract a Google Sheets ID from GOOGLE_SHEETS_URL.\n"
            "Make sure it looks like: https://docs.google.com/spreadsheets/d/YOUR_ID/edit"
        )
    sheet_id = match.group(1)

    # Extract optional gid (tab identifier)
    gid_match = re.search(r"gid=(\d+)", url)
    gid = gid_match.group(1) if gid_match else "0"

    return (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        f"/export?format=csv&gid={gid}"
    )


def fetch_students(sheets_url: str) -> list[tuple[str, str, str]]:
    """
    Download the Google Sheet as CSV and return a list of
    (name, superlative, drawing_instructions) tuples.
    The sheet is expected to have three columns with headers:
      name | superlative | drawing_instructions
    The first row is treated as a header and skipped.
    """
    csv_url = sheet_url_to_csv_url(sheets_url)
    print(f"  Fetching sheet from: {csv_url}")

    response = requests.get(csv_url, timeout=30)
    if response.status_code != 200:
        sys.exit(
            f"ERROR: Failed to download Google Sheet (HTTP {response.status_code}).\n"
            "Make sure the sheet is shared as 'Anyone with the link can view'."
        )

    reader = csv.reader(io.StringIO(response.text))
    rows = list(reader)

    if len(rows) < 2:
        sys.exit("ERROR: The sheet appears to be empty or has only a header row.")

    # Skip the header row; expect columns: name, superlative, drawing_instructions
    students = []
    for i, row in enumerate(rows[1:], start=2):
        if len(row) < 3:
            print(f"  WARNING: Row {i} has fewer than 3 columns — skipping.")
            continue
        name = row[0].strip()
        superlative = row[1].strip()
        drawing_instructions = row[2].strip()
        
        # Check for required fields
        if not name or not superlative:
            print(f"  WARNING: Row {i} is missing name or superlative — skipping.")
            continue
        
        # If drawing_instructions is empty, try to load from file
        if not drawing_instructions:
            print(f"  INFO: Row {i} ({name}) has no drawing_instructions, checking photos directory...")
            drawing_instructions = load_drawing_instructions_from_file(name)
            if drawing_instructions:
                print(f"  ✓ Loaded drawing instructions from photos/{normalize_name_for_file(name)}.txt")
            else:
                print(f"  WARNING: Row {i} ({name}) has no drawing_instructions and no matching file found — skipping.")
                continue
        
        students.append((name, superlative, drawing_instructions))

    print(f"  Found {len(students)} student(s).")
    return students


def sanitize_filename(name: str) -> str:
    """Convert a student name to a safe filename (no extension)."""
    # Replace spaces with underscores, remove characters that aren't alphanumeric/underscore/hyphen
    safe = re.sub(r"[^\w\s-]", "", name).strip()
    safe = re.sub(r"[\s]+", "_", safe)
    return safe.lower()


def build_prompt(prompt_intro: str, superlative: str, drawing_instructions: str) -> str:
    """Combine the style guide intro with the superlative and drawing instructions."""
    return (
        f"{prompt_intro}\n\n"
        f"---\n\n"
        f"SUPERLATIVE: {superlative}\n\n"
        f"DRAWING INSTRUCTIONS: {drawing_instructions}"
    )


PHOTO_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

def find_photo(name: str) -> str | None:
    """
    Look for a reference photo in the photos/ directory matching the student's name.
    Uses the same filename normalization as the .txt file lookup.
    Returns the full path to the photo if found, None otherwise.
    """
    normalized = normalize_name_for_file(name)
    for ext in PHOTO_EXTENSIONS:
        path = os.path.join(PHOTOS_DIR, normalized + ext)
        if os.path.exists(path):
            return path
    return None


def _encode_photo(photo_path: str) -> tuple[str, str]:
    """Read a photo and return (base64_string, mime_type)."""
    ext = os.path.splitext(photo_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(ext, "image/jpeg")
    with open(photo_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("utf-8")
    return b64, mime_type


def generate_and_save_image(
    client: OpenAI,
    prompt: str,
    output_path: str,
    name: str,
    image_model: str = DEFAULT_IMAGE_MODEL,
    photo_path: str | None = None,
) -> bool:
    """
    Call the OpenAI image generation API and save the result to output_path.
    If photo_path is provided, sends the reference photo alongside the prompt
    using the Responses API (multi-modal input).
    Prints elapsed seconds in-place while waiting for the API response.
    Returns True on success, False on failure.
    """
    start_time = time.time()
    stop_event = threading.Event()

    def _timer():
        """Background thread: update elapsed time on the same line every second."""
        while not stop_event.wait(1):
            elapsed = int(time.time() - start_time)
            print(f"\r  ⏳  Generating...  {elapsed}s ", end="", flush=True)

    timer_thread = threading.Thread(target=_timer, daemon=True)
    timer_thread.start()

    try:
        if photo_path:
            # ── Multi-modal: text prompt + reference photo via Responses API ──
            b64, mime_type = _encode_photo(photo_path)
            full_prompt = (
                f"REFERENCE PHOTO: The image attached is a real photo of the student. "
                f"Use it to capture their likeness in the cartoon.\n\n{prompt}"
            )
            response = client.responses.create(
                model=image_model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:{mime_type};base64,{b64}",
                            },
                            {
                                "type": "input_text",
                                "text": full_prompt,
                            },
                        ],
                    }
                ],
            )
            # Extract base64 image from the Responses API output
            image_data = None
            for item in response.output:
                if hasattr(item, "type") and item.type == "image_generation_call":
                    image_data = item.result
                    break
            if image_data is None:
                raise ValueError(
                    f"No image_generation_call found in response output. "
                    f"Output types: {[getattr(o, 'type', '?') for o in response.output]}"
                )
        else:
            # ── Text-only: standard Images API ──
            response = client.images.generate(
                model=image_model,
                prompt=prompt,
                size=IMAGE_SIZE,
                quality=IMAGE_QUALITY,
                n=1,
            )
            image_data = response.data[0].b64_json

        stop_event.set()
        elapsed = int(time.time() - start_time)

        image_bytes = base64.b64decode(image_data)
        with open(output_path, "wb") as f:
            f.write(image_bytes)

        # Overwrite the timer line with the success message
        print(f"\r  ✅  Saved: {os.path.basename(output_path)}  ({elapsed}s)          ")
        return True

    except Exception as e:
        stop_event.set()
        elapsed = int(time.time() - start_time)
        # Move past the timer line, then print the full error
        print(f"\r  ❌  Failed for '{name}':  ({elapsed}s)                  ")
        print(f"      {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Paper Plate Awards — Image Generator")
    print("=" * 60)

    # 1. Load config & prompt intro
    print("\n[1/4] Loading configuration…")
    config = load_config(CONFIG_PATH)
    prompt_intro = load_prompt_intro(PROMPT_INTRO_PATH)
    print("  Config and prompt intro loaded.")

    # 2. Fetch student data from Google Sheets
    print("\n[2/4] Fetching student data from Google Sheets…")
    students = fetch_students(config["GOOGLE_SHEETS_URL"])

    # 3. Set up output directory and OpenAI client
    print("\n[3/4] Setting up output directory and OpenAI client…")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = OpenAI(api_key=config["OPENAI_API_KEY"])
    image_model = config.get("IMAGE_MODEL", DEFAULT_IMAGE_MODEL)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Model: {image_model}  |  Size: {IMAGE_SIZE}  |  Quality: {IMAGE_QUALITY}")

    # 4. Generate images
    print(f"\n[4/4] Generating {len(students)} image(s)…\n")
    successes = 0
    skipped = 0
    failures = 0

    for i, (name, superlative, drawing_instructions) in enumerate(students, start=1):
        filename = sanitize_filename(name) + ".png"
        output_path = os.path.join(OUTPUT_DIR, filename)

        print(f"[{i}/{len(students)}] {name}")

        # Skip if already generated
        if os.path.exists(output_path):
            print(f"  ⏭️  Already exists — skipping. (Delete file to regenerate.)")
            skipped += 1
            continue

        photo = find_photo(name)
        if photo:
            print(f"  📷  Reference photo: {os.path.basename(photo)}")
        prompt = build_prompt(prompt_intro, superlative, drawing_instructions)
        success = generate_and_save_image(client, prompt, output_path, name, image_model, photo)

        if success:
            successes += 1
        else:
            failures += 1

        # Polite delay between requests (skip after last one)
        if i < len(students):
            time.sleep(INTER_REQUEST_DELAY)

    # Summary
    print("\n" + "=" * 60)
    print(f"  Done!  ✅ {successes} generated  |  ⏭️ {skipped} skipped  |  ❌ {failures} failed")
    print(f"  Images saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
