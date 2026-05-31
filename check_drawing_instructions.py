#!/usr/bin/env python3
"""
Paper Plate Awards - Drawing Instructions Checker
Validates that every person in the Google Sheet has drawing instructions
either specified in the sheet OR available in a corresponding text file.
"""

import csv
import io
import os
import re
import sys

import requests
import yaml

# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")
PHOTOS_DIR = os.path.join(SCRIPT_DIR, "photos")


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

    if not config.get("GOOGLE_SHEETS_URL"):
        sys.exit(
            "ERROR: 'GOOGLE_SHEETS_URL' is not set in config.yaml.\n"
            "Please fill in your Google Sheets URL before running."
        )
    return config


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


def sheet_url_to_csv_url(url: str) -> str:
    """
    Convert a Google Sheets edit/view URL to a CSV export URL.
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


def check_file_exists(name: str) -> bool:
    """Check if a drawing instructions file exists for the given name."""
    normalized_name = normalize_name_for_file(name)
    file_path = os.path.join(PHOTOS_DIR, f"{normalized_name}.txt")
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0


def main():
    print("=" * 70)
    print("  Paper Plate Awards — Drawing Instructions Checker")
    print("=" * 70)

    # Load config
    print("\n[1/2] Loading configuration…")
    config = load_config(CONFIG_PATH)
    print("  Config loaded.")

    # Fetch sheet data
    print("\n[2/2] Fetching and validating student data…")
    csv_url = sheet_url_to_csv_url(config["GOOGLE_SHEETS_URL"])
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

    print(f"  Found {len(rows) - 1} row(s) in the sheet.\n")
    print("=" * 70)

    # Track results
    valid_count = 0
    missing_count = 0
    missing_entries = []

    # Check each row
    for i, row in enumerate(rows[1:], start=2):
        if len(row) < 3:
            print(f"⚠️  Row {i}: Has fewer than 3 columns — SKIPPING")
            continue

        name = row[0].strip()
        superlative = row[1].strip()
        drawing_instructions = row[2].strip()

        if not name or not superlative:
            print(f"⚠️  Row {i}: Missing name or superlative — SKIPPING")
            continue

        # Check if drawing instructions exist
        has_sheet_instructions = bool(drawing_instructions)
        has_file_instructions = check_file_exists(name)

        if has_sheet_instructions:
            print(f"✅ Row {i}: {name:30} → Has instructions in sheet")
            valid_count += 1
        elif has_file_instructions:
            normalized = normalize_name_for_file(name)
            print(f"✅ Row {i}: {name:30} → Has file: photos/{normalized}.txt")
            valid_count += 1
        else:
            normalized = normalize_name_for_file(name)
            print(f"❌ Row {i}: {name:30} → MISSING (no sheet data, no file: photos/{normalized}.txt)")
            missing_count += 1
            missing_entries.append((i, name, normalized))

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  ✅ Valid entries:   {valid_count}")
    print(f"  ❌ Missing entries: {missing_count}")

    if missing_count > 0:
        print("\n" + "=" * 70)
        print("  MISSING DRAWING INSTRUCTIONS")
        print("=" * 70)
        for row_num, name, normalized in missing_entries:
            print(f"  • Row {row_num}: {name}")
            print(f"    Expected file: photos/{normalized}.txt")
        print("\n  Please add drawing instructions to the Google Sheet OR create")
        print("  the corresponding .txt files in the photos directory.")
        print("=" * 70)
        sys.exit(1)
    else:
        print("\n  🎉 All entries have drawing instructions!")
        print("=" * 70)


if __name__ == "__main__":
    main()
