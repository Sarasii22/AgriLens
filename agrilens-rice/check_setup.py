"""Run this once to check your setup:  python check_setup.py

It confirms that your API key works, lists models you can use,
and makes one tiny test call with the model AgriLens will use.
"""
from engine import DEFAULT_MODEL, make_client, read_setting


def main():
    key = read_setting("GEMINI_API_KEY")
    if not key or "paste" in key.lower():
        raise SystemExit(
            "No real GEMINI_API_KEY found.\n"
            "Copy .streamlit/secrets.toml.example to .streamlit/secrets.toml and paste your key."
        )
    print("API key found.")

    client = make_client(key)
    model = read_setting("GEMINI_MODEL", DEFAULT_MODEL)

    print("\nModels available to your key (first 30):")
    shown = 0
    try:
        for item in client.models.list():
            name = getattr(item, "name", "")
            actions = getattr(item, "supported_actions", None) or []
            if "gemini" in name and (not actions or "generateContent" in actions):
                print("  ", name.replace("models/", ""))
                shown += 1
                if shown >= 30:
                    break
    except Exception as error:  # noqa: BLE001
        print("  Could not list models:", error)

    print(f"\nTesting model '{model}' ...")
    try:
        response = client.models.generate_content(
            model=model, contents="Reply with the single word: ready"
        )
        print("Model replied:", (response.text or "").strip())
        print("\nAll good. You can run:  streamlit run app.py")
    except Exception as error:  # noqa: BLE001
        print("Test call failed:", error)
        print(
            "\nIf the model was not found, pick a name from the list above and set it\n"
            'in .streamlit/secrets.toml like:  GEMINI_MODEL = "model-name-here"'
        )


if __name__ == "__main__":
    main()
