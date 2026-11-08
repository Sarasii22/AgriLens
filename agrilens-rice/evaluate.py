"""Measure how accurate AgriLens is on YOUR test photos.

Put photos in folders named after the correct answer:

    test_images/
        leaf_blast/        photo1.jpg photo2.jpg ...
        neck_blast/
        brown_spot/
        narrow_brown_spot/
        hispa/
        healthy/
        not_rice/          photos that are NOT rice (the app should say not_rice)

Run:  python evaluate.py
Results are printed and saved to results.csv (use them in your Devpost write-up).
"""
import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path

from engine import DEFAULT_MODEL, diagnose, make_client, predicted_label, prepare_image, read_setting

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="test_images", help="folder with one sub-folder per label")
    parser.add_argument("--delay", type=float, default=6.0, help="seconds to wait between photos (free-tier limits)")
    args = parser.parse_args()

    api_key = read_setting("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("No GEMINI_API_KEY found. Put it in .streamlit/secrets.toml first.")
    model = read_setting("GEMINI_MODEL", DEFAULT_MODEL)
    client = make_client(api_key)

    root = Path(args.folder)
    if not root.exists():
        raise SystemExit(f"Folder '{root}' not found.")

    rows = []
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        expected = label_dir.name
        for image_path in sorted(label_dir.iterdir()):
            if image_path.suffix.lower() not in EXTENSIONS:
                continue
            try:
                image_bytes, mime = prepare_image(image_path.read_bytes())
                result = diagnose(client, model, image_bytes, mime)
                predicted = predicted_label(result)
                confidence = result["confidence"]
                quality = result["photo_quality"]
            except Exception as error:  # noqa: BLE001
                predicted, confidence, quality = "error", "", ""
                print(f"  ERROR on {image_path.name}: {error}")
            mark = "OK " if predicted == expected else "BAD"
            print(f"[{mark}] {expected:18s} -> {predicted:18s} ({confidence}) {image_path.name}")
            rows.append(
                {
                    "file": str(image_path),
                    "expected": expected,
                    "predicted": predicted,
                    "confidence": confidence,
                    "photo_quality": quality,
                }
            )
            time.sleep(args.delay)

    if not rows:
        raise SystemExit("No images found. Add photos to the sub-folders first.")

    with open("results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    correct = sum(r["predicted"] == r["expected"] for r in rows)
    unsure = sum(r["predicted"] in ("unsure", "error") for r in rows)
    confident = [r for r in rows if r["predicted"] not in ("unsure", "error") and r["confidence"] != "low"]
    confident_correct = sum(r["predicted"] == r["expected"] for r in confident)

    print("\n=== SUMMARY ===")
    print(f"Photos tested: {total}")
    print(f"Overall accuracy: {correct}/{total} = {100 * correct / total:.0f}%")
    print(f"'Unsure' or error answers: {unsure}")
    if confident:
        print(
            f"When confident (not low): {confident_correct}/{len(confident)} = "
            f"{100 * confident_correct / len(confident):.0f}%"
        )

    per_class = defaultdict(lambda: [0, 0])
    for row in rows:
        per_class[row["expected"]][1] += 1
        per_class[row["expected"]][0] += row["predicted"] == row["expected"]
    print("\nPer label (correct/total):")
    for label, (right, count) in sorted(per_class.items()):
        print(f"  {label:18s} {right}/{count}")

    mistakes = Counter((r["expected"], r["predicted"]) for r in rows if r["predicted"] != r["expected"])
    if mistakes:
        print("\nMost common mix-ups (expected -> predicted):")
        for (expected, predicted), count in mistakes.most_common(8):
            print(f"  {expected} -> {predicted}: {count}")
    print("\nSaved details to results.csv")


if __name__ == "__main__":
    main()
