"""AgriLens engine.

Does the real work for the app and for the evaluation script:
  1. prepare_image()   shrinks the photo so it is quick and cheap to send
  2. diagnose()        asks Gemini to choose ONE label from a fixed list
  3. load_cards()      reads the disease cards (our verified knowledge base)
  4. advice_from_card()  writes advice using ONLY the matching card
  5. general_advice()  careful fallback when the app is not sure
"""
import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
CARDS_DIR = BASE_DIR / "cards"

# Check which models your free key can use (run check_setup.py) and change if needed.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

LANGUAGES = ["English", "Sinhala", "Tamil"]

# The fixed list of answers the AI is allowed to give. Each one has a card in cards/.
DISEASE_NAMES = {
    "leaf_blast": "Rice leaf blast",
    "neck_blast": "Rice neck blast",
    "brown_spot": "Rice brown spot",
    "narrow_brown_spot": "Rice narrow brown spot",
    "hispa": "Rice hispa (insect damage)",
    "healthy": "Healthy rice plant",
}
VALID_LABELS = set(DISEASE_NAMES) | {"unsure"}

SAFETY_NOTE = (
    "AgriLens gives general guidance, not a certain diagnosis. "
    "If the problem is spreading or serious, show the plant to your nearest "
    "Department of Agriculture or agrarian services officer before spending money on chemicals."
)


# ---------------------------------------------------------------- settings
def read_setting(name, default=None):
    """Read a setting from an environment variable, or from .streamlit/secrets.toml."""
    value = os.getenv(name)
    if value:
        return value.strip()
    secrets_file = BASE_DIR / ".streamlit" / "secrets.toml"
    if secrets_file.exists():
        text = secrets_file.read_text(encoding="utf-8")
        match = re.search(
            rf'^\s*{name}\s*=\s*["\']([^"\']+)["\']', text, flags=re.MULTILINE
        )
        if match:
            return match.group(1).strip()
    return default


def make_client(api_key):
    return genai.Client(api_key=api_key)


# ------------------------------------------------------------------ images
def prepare_image(file_bytes, max_side=1024):
    """Fix phone rotation, convert to JPEG, and shrink. Returns (bytes, mime_type)."""
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=88)
    return buffer.getvalue(), "image/jpeg"


# ------------------------------------------------------------ API helpers
def _with_retry(fn, tries=4):
    """Free tiers sometimes say 'busy' (429/503). Wait a little and try again."""
    delay = 4
    for attempt in range(tries):
        try:
            return fn()
        except Exception as error:  # noqa: BLE001
            message = str(error)
            transient = any(
                code in message
                for code in ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "overloaded")
            )
            if not transient or attempt == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def _generate_text(client, model, system_prompt, contents, temperature=0.3):
    def call():
        return client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
            ),
        )

    response = _with_retry(call)
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("The AI returned an empty answer. Please try again.")
    return text


# --------------------------------------------------------------- diagnosis
class Diagnosis(BaseModel):
    is_rice_plant: bool = Field(
        description="True only if the photo clearly shows a rice plant (leaf, stem or grain head)."
    )
    photo_quality: str = Field(description="Either 'good' or 'poor'.")
    disease: str = Field(
        description="One of: leaf_blast, neck_blast, brown_spot, narrow_brown_spot, hispa, healthy, unsure."
    )
    confidence: str = Field(description="One of: low, medium, high.")
    visible_symptoms: str = Field(description="One or two plain sentences describing what is visible.")
    reasoning: str = Field(description="One sentence explaining the choice.")


DIAGNOSE_PROMPT = """You are a rice plant problem identification assistant for a farmer-support app in Sri Lanka.
You receive ONE photo. Decide what it shows, using ONLY the labels below.

Labels for "disease":
- leaf_blast: spindle-shaped leaf spots (wide in the middle, pointed ends) with brown or reddish-brown edges and grey or ash-coloured centres.
- neck_blast: the neck just below the grain head (panicle) is brown or black; the grain head may bend, break or look bleached with empty grains.
- brown_spot: small dark brown spots up to large oval or round spots with a dark brown edge and a lighter reddish-brown or grey centre; may also darken the grains.
- narrow_brown_spot: short, narrow, straight brown lines running parallel to the leaf veins.
- hispa: insect damage; white or silvery parallel streaks or scraped patches on the leaf surface, sometimes pale blisters, sometimes small spiny dark beetles.
- healthy: evenly green plant with no spots, streaks or discolouration.
- unsure: use this when none of the above clearly fits, the symptoms are unclear, several labels are equally likely, or the problem is something else (for example a nutrient problem).

Rules:
- Set is_rice_plant to false if the photo does not clearly show a rice plant.
- Set photo_quality to "poor" if the photo is too blurry, too dark, too far away, or the affected part is not visible. Otherwise "good".
- Never guess. If you are not sure, choose "unsure" with confidence "low".
- Use confidence "high" only when the typical symptoms are clearly visible.
- visible_symptoms: describe only what you can actually see.
- Return only the JSON object."""


def _extract_json(response):
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, BaseModel):
            return parsed.model_dump()
        if isinstance(parsed, dict):
            return parsed
    text = (response.text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


def _clean_diagnosis(data):
    """Never trust the AI blindly: force every field into a safe, known value."""
    disease = str(data.get("disease", "unsure")).strip().lower().replace(" ", "_")
    if disease not in VALID_LABELS:
        disease = "unsure"

    confidence = str(data.get("confidence", "low")).strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"

    quality = "good" if str(data.get("photo_quality", "poor")).strip().lower() == "good" else "poor"

    is_rice = data.get("is_rice_plant", False)
    if isinstance(is_rice, str):
        is_rice = is_rice.strip().lower() == "true"

    return {
        "is_rice_plant": bool(is_rice),
        "photo_quality": quality,
        "disease": disease,
        "confidence": confidence,
        "visible_symptoms": str(data.get("visible_symptoms", "")).strip(),
        "reasoning": str(data.get("reasoning", "")).strip(),
    }


def diagnose(client, model, image_bytes, mime_type="image/jpeg"):
    """Step 1 of the pipeline: look at the photo and pick a label from the fixed list."""
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    def call():
        return client.models.generate_content(
            model=model,
            contents=[image_part, "Classify this photo."],
            config=types.GenerateContentConfig(
                system_instruction=DIAGNOSE_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=Diagnosis,
            ),
        )

    response = _with_retry(call)
    return _clean_diagnosis(_extract_json(response))


def predicted_label(diagnosis):
    """One simple label per photo, used by the evaluation script."""
    if not diagnosis["is_rice_plant"]:
        return "not_rice"
    return diagnosis["disease"]


# ------------------------------------------------------------------- cards
@dataclass
class Card:
    id: str
    name: str
    verified: bool
    source_name: str
    source_url: str
    accessed: str
    body: str


def parse_card(text):
    """Cards are Markdown files with a small 'key: value' header between --- lines."""
    meta = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        header, body = parts[1], parts[2]
        for line in header.strip().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
    return meta, body.strip()


def load_cards():
    """Read every card in cards/ and index it by its id."""
    cards = {}
    for path in sorted(CARDS_DIR.glob("*.md")):
        meta, body = parse_card(path.read_text(encoding="utf-8"))
        card_id = meta.get("id")
        if not card_id:
            continue
        cards[card_id] = Card(
            id=card_id,
            name=meta.get("name", card_id),
            verified=meta.get("verified", "no").lower() == "yes",
            source_name=meta.get("source_name", ""),
            source_url=meta.get("source_url", ""),
            accessed=meta.get("accessed", ""),
            body=body,
        )
    return cards


# ------------------------------------------------------------------ advice
CARD_ADVICE_PROMPT = """You are AgriLens, helping a small-scale rice farmer in Sri Lanka.
Write advice about "{disease_name}" using ONLY the reference card below.

Rules:
- Use only facts that are in the card. If something is not in the card, do not add it.
- Never name specific pesticides, brands or doses. If chemical control matters, tell the farmer to ask the nearest Department of Agriculture or agrarian services officer.
- Use simple, short sentences and no technical jargon.
- Reply entirely in {language}, including the headings.
- Use these headings (adapt them if the plant is healthy): What it is / How to recognise it / What you can do now / When to get help.
- Keep it under 200 words.

REFERENCE CARD:
{card_body}
"""

GENERAL_ADVICE_PROMPT = """You are AgriLens, helping a small-scale rice farmer in Sri Lanka.
The photo may show a rice plant problem, but the app could not identify it with confidence.
Give cautious, general guidance only.

Rules:
- Start with one or two simple sentences on what you can see, and say clearly that you are not sure what the problem is.
- Ask for a clearer, closer, well-lit photo of one affected leaf or grain head.
- Give only safe, general good-practice steps (check the field, remove a few badly affected leaves, avoid too much nitrogen fertiliser, keep weeds down, check water management).
- Do not state any disease as certain. Do not name pesticides, brands or doses.
- Tell the farmer to show the plant to the nearest Department of Agriculture or agrarian services officer.
- Use simple, short sentences. Reply entirely in {language}. Keep it under 150 words.
"""


def advice_from_card(client, model, disease_name, card_body, language):
    """Step 3 of the pipeline: write advice grounded in the card (this is the 'RAG' step)."""
    system_prompt = CARD_ADVICE_PROMPT.format(
        disease_name=disease_name, language=language, card_body=card_body
    )
    return _generate_text(client, model, system_prompt, "Write the advice now.", temperature=0.2)


def general_advice(client, model, image_bytes, mime_type, language):
    """Fallback when the app is not sure. Always shown as unverified."""
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    system_prompt = GENERAL_ADVICE_PROMPT.format(language=language)
    return _generate_text(client, model, system_prompt, [image_part, "Please help."], temperature=0.3)
