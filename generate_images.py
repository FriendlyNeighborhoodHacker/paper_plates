#!/usr/bin/env python3
"""
Paper Plate Awards - Image Generator
Fetches student names and superlative descriptions from Google Sheets,
then generates paper plate award images using OpenAI's image generation API.
"""

import base64
import csv
import io
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
import yaml
from openai import OpenAI

# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")
PROMPT_INTRO_PATH = os.path.join(SCRIPT_DIR, "prompt_intro.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "generated_images")
PHOTOS_DIR = os.path.join(SCRIPT_DIR, "photos")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")

IMAGE_SIZE = "1024x1024"      # Square — ideal for a circular plate composition
IMAGE_QUALITY = "high"        # Best quality available on the plan

# Seconds to wait between API calls to be polite to rate limits
INTER_REQUEST_DELAY = 3

# Default model if not specified in config
DEFAULT_IMAGE_MODEL = "gpt-image-1"

# Module-level logger (configured in main())
logger = logging.getLogger("paper_plates")


def setup_logging() -> str:
    """
    Configure logging to write to both the terminal and a timestamped log file.
    Terminal: no timestamp (clean live output).
    File: timestamp + level + message (permanent record).
    Returns the log file path.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(LOGS_DIR, f"generate_{run_ts}.log")

    logger.setLevel(logging.DEBUG)

    # Terminal handler — clean, no timestamp
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    # File handler — timestamped
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-5s  %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return log_path


# Thread-safe log (logging module is already thread-safe)
_safe_print = logger.info


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
        if not name:
            print(f"  WARNING: Row {i} has no name — skipping.")
            continue
        if not superlative:
            print(f"  WARNING: Row {i} ({name}) has no superlative — skipping.")
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


def generate_and_save_image(
    client: OpenAI,
    prompt: str,
    output_path: str,
    name: str,
    image_model: str = DEFAULT_IMAGE_MODEL,
    photo_path: str | None = None,
    label: str = "",
) -> bool:
    """
    Call the OpenAI image generation API and save the result to output_path.
    If photo_path is provided, sends the reference photo via images.edit().
    If label is provided (parallel mode), prints prefixed lines instead of \r timer.
    Returns True on success, False on failure.
    """
    start_time = time.time()
    parallel = bool(label)
    stop_event = threading.Event()

    logger.info(f"Starting to generate image for {name}")

    if not parallel:
        # Sequential mode: in-place \r timer (updates every second)
        def _timer():
            while not stop_event.wait(1):
                elapsed = int(time.time() - start_time)
                print(f"\r  ⏳  Generating...  {elapsed}s ", end="", flush=True)
    else:
        # Parallel mode: log a heartbeat every 10 seconds
        def _timer():
            interval = 10
            while not stop_event.wait(interval):
                elapsed = int(time.time() - start_time)
                logger.info(f"{label} ⏳  {elapsed}s...")

    timer_thread = threading.Thread(target=_timer, daemon=True)
    timer_thread.start()

    try:
        if photo_path:
            full_prompt = (
                f"REFERENCE PHOTO: The attached image is a real photo of the student. "
                f"Use it to capture their likeness in the cartoon.\n\n{prompt}"
            )
            logger.info(f"Sending prompt for {name} (+ reference photo {os.path.basename(photo_path)}):\n{full_prompt}")
            with open(photo_path, "rb") as photo_file:
                response = client.images.edit(
                    model=image_model,
                    image=photo_file,
                    prompt=full_prompt,
                    size=IMAGE_SIZE,
                    quality=IMAGE_QUALITY,
                    n=1,
                )
        else:
            logger.info(f"Sending prompt for {name}:\n{prompt}")
            response = client.images.generate(
                model=image_model,
                prompt=prompt,
                size=IMAGE_SIZE,
                quality=IMAGE_QUALITY,
                n=1,
            )

        image_data = response.data[0].b64_json
        elapsed = int(time.time() - start_time)

        image_bytes = base64.b64decode(image_data)
        with open(output_path, "wb") as f:
            f.write(image_bytes)

        stop_event.set()
        logger.info(f"Generated image successfully for {name} ({elapsed}s)  →  {os.path.basename(output_path)}")
        if not parallel:
            print(f"\r  ✅  Saved: {os.path.basename(output_path)}  ({elapsed}s)          ")
        return True

    except Exception as e:
        elapsed = int(time.time() - start_time)
        stop_event.set()
        logger.error(f"Failed to generate image for {name} ({elapsed}s): {e}")
        if not parallel:
            print(f"\r")  # move past the timer line
        return False


# ── Worker ───────────────────────────────────────────────────────────────────

def _process_student(
    i: int,
    total: int,
    name: str,
    superlative: str,
    drawing_instructions: str,
    client: OpenAI,
    prompt_intro: str,
    image_model: str,
    parallel: bool,
) -> str:
    """
    Process one student: build prompt, optionally find reference photo,
    call the API, and save the image.
    Returns "success", "skipped", or "failed".
    """
    filename = sanitize_filename(name) + ".png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    label = f"[{i}/{total}] {name}:"

    if os.path.exists(output_path):
        logger.info(f"{label} ⏭️  Already exists — skipping.")
        return "skipped"

    photo = find_photo(name)
    prompt = build_prompt(prompt_intro, superlative, drawing_instructions)

    # Build the display version of the prompt
    if photo:
        full_display = (
            f"REFERENCE PHOTO: The attached image is a real photo of the student. "
            f"Use it to capture their likeness in the cartoon.\n\n{prompt}"
        )
    else:
        full_display = prompt

    # Print the prompt block (thread-safe)
    lines = []
    lines.append(f"\n{label}")
    if photo:
        lines.append(f"  📷  Reference photo: {os.path.basename(photo)}")
    lines.append(f"  ── Prompt {'─' * 42}")
    if photo:
        lines.append(f"  [+ reference photo: {os.path.basename(photo)}]")
    for line in full_display.splitlines():
        lines.append(f"  {line}")
    lines.append(f"  {'─' * 50}")
    if parallel:
        lines.append(f"  ⏳  Generating...")
    logger.info("\n".join(lines))

    success = generate_and_save_image(
        client=client,
        prompt=prompt,
        output_path=output_path,
        name=name,
        image_model=image_model,
        photo_path=photo,
        label=label if parallel else "",
    )
    return "success" if success else "failed"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Paper Plate Awards — Image Generator")
    print("=" * 60)

    # 0. Set up logging
    log_path = setup_logging()

    # 1. Load config & prompt intro
    logger.info("\n[1/4] Loading configuration…")
    config = load_config(CONFIG_PATH)
    prompt_intro = load_prompt_intro(PROMPT_INTRO_PATH)
    workers = int(config.get("PARALLEL_WORKERS", 1))
    logger.info(f"  Config and prompt intro loaded.")
    logger.info(f"  Log file: {log_path}")

    # 2. Fetch student data from Google Sheets
    logger.info("\n[2/4] Fetching student data from Google Sheets…")
    students = fetch_students(config["GOOGLE_SHEETS_URL"])

    # 3. Set up output directory and OpenAI client
    logger.info("\n[3/4] Setting up output directory and OpenAI client…")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = OpenAI(api_key=config["OPENAI_API_KEY"])
    image_model = config.get("IMAGE_MODEL", DEFAULT_IMAGE_MODEL)
    logger.info(f"  Output directory: {OUTPUT_DIR}")
    logger.info(f"  Model: {image_model}  |  Size: {IMAGE_SIZE}  |  Quality: {IMAGE_QUALITY}")
    logger.info(f"  Parallel workers: {workers}")

    # 4. Generate images
    total = len(students)
    logger.info(f"\n[4/4] Generating {total} image(s) with {workers} worker(s)…\n")
    successes = 0
    skipped = 0
    failures = 0
    failed_names: list[str] = []

    if workers <= 1:
        # ── Sequential mode ──────────────────────────────────────
        for i, (name, superlative, drawing_instructions) in enumerate(students, start=1):
            result = _process_student(
                i, total, name, superlative, drawing_instructions,
                client, prompt_intro, image_model, parallel=False,
            )
            if result == "success":
                successes += 1
            elif result == "skipped":
                skipped += 1
            else:
                failures += 1
                failed_names.append(name)
            if i < total and result != "skipped":
                time.sleep(INTER_REQUEST_DELAY)

    else:
        # ── Parallel mode ────────────────────────────────────────
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for i, (name, superlative, drawing_instructions) in enumerate(students, start=1):
                future = executor.submit(
                    _process_student,
                    i, total, name, superlative, drawing_instructions,
                    client, prompt_intro, image_model, True,
                )
                futures[future] = name

            for future in as_completed(futures):
                name = futures[future]
                result = future.result()
                if result == "success":
                    successes += 1
                elif result == "skipped":
                    skipped += 1
                else:
                    failures += 1
                    failed_names.append(name)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"  Done!  ✅ {successes} generated  |  ⏭️ {skipped} skipped  |  ❌ {failures} failed")
    if failed_names:
        logger.info(f"\n  Failed students:")
        for fn in sorted(failed_names):
            logger.info(f"    • {fn}")
    logger.info(f"\n  Images saved to: {OUTPUT_DIR}")
    logger.info(f"  Log saved to:    {log_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
