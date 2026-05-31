#!/usr/bin/env python3
"""
test_openai_key.py — Validates that the OPENAI_API_KEY in config.yaml is valid.

Makes a lightweight API call (listing models) to confirm the key is accepted
by OpenAI. No images are generated and no significant cost is incurred.

Usage:
    python test_openai_key.py
"""

from openai import OpenAI, AuthenticationError, APIConnectionError

from generate_images import CONFIG_PATH, DEFAULT_IMAGE_MODEL, load_config


def main():
    print("=" * 60)
    print("  Paper Plate Awards — OpenAI Key Test")
    print("=" * 60)

    # Load config (validates that OPENAI_API_KEY is present)
    print("\nLoading config…")
    config = load_config(CONFIG_PATH)
    api_key = config["OPENAI_API_KEY"]
    masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    print(f"  API key loaded: {masked}")

    # Create client and make a lightweight API call
    print("\nContacting OpenAI API…")
    client = OpenAI(api_key=api_key)

    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]

        # Check that our configured image model is available
        target_model = config.get("IMAGE_MODEL", DEFAULT_IMAGE_MODEL)
        if target_model in model_ids:
            status = f"✅  '{target_model}' is available on your account."
        else:
            status = (
                f"⚠️  '{target_model}' was NOT found in your model list.\n"
                "    Your key is valid, but you may not have access to this model.\n"
                "    Make sure you are on the $100/month plan."
            )

        print("\n" + "=" * 60)
        print("  ✅  API key is VALID — OpenAI accepted it successfully.")
        print(f"  {status}")
        print(f"  Total models available: {len(model_ids)}")
        print("=" * 60)

    except AuthenticationError as e:
        print("\n" + "=" * 60)
        print("  ❌  API key is INVALID — OpenAI rejected it.")
        print(f"      Error: {e}")
        print("  Check your OPENAI_API_KEY in config.yaml.")
        print("=" * 60)

    except APIConnectionError as e:
        print("\n" + "=" * 60)
        print("  ❌  Could not connect to OpenAI — check your internet connection.")
        print(f"      Error: {e}")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"  ❌  Unexpected error: {e}")
        print("=" * 60)


if __name__ == "__main__":
    main()
