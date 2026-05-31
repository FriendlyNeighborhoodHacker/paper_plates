#!/usr/bin/env python3
"""
test_sheets.py — Standalone test for the Google Sheets fetch step.

Imports shared functions from generate_images.py and prints the parsed
student data without making any OpenAI API calls.

Usage:
    python test_sheets.py
"""

import sys
import yaml

from generate_images import CONFIG_PATH, fetch_students


def load_sheets_config(path: str) -> dict:
    """Load config and validate only GOOGLE_SHEETS_URL (no API key needed for this test)."""
    import os
    if not os.path.exists(path):
        sys.exit(f"ERROR: Config file not found at '{path}'.")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    url = config.get("GOOGLE_SHEETS_URL", "")
    if not url or url.startswith("https://docs.google.com/spreadsheets/d/YOUR"):
        sys.exit(
            "ERROR: 'GOOGLE_SHEETS_URL' is not set in config.yaml.\n"
            "Please fill in your Google Sheets URL before running."
        )
    return config


def main():
    print("=" * 60)
    print("  Paper Plate Awards — Sheet Fetch Test")
    print("=" * 60)

    # Load config — only checks GOOGLE_SHEETS_URL, no API key required
    print("\nLoading config…")
    config = load_sheets_config(CONFIG_PATH)
    print("  Config loaded.")

    # Fetch and parse the sheet
    print("\nFetching student data from Google Sheets…")
    students = fetch_students(config["GOOGLE_SHEETS_URL"])

    # Print results
    print("\n" + "-" * 60)
    for i, (name, superlative, drawing_instructions) in enumerate(students, start=1):
        print(f"\n[{i}] {name}")
        print(f"    Superlative:          {superlative}")
        print(f"    Drawing Instructions: {drawing_instructions}")

    print("\n" + "=" * 60)
    print(f"  Total: {len(students)} student(s) fetched successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
