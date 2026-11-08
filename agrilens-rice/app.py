"""AgriLens: rice problem helper (Streamlit app).

Run locally with:  streamlit run app.py
"""
import hashlib

import streamlit as st

from engine import (
    DEFAULT_MODEL,
    DISEASE_NAMES,
    LANGUAGES,
    SAFETY_NOTE,
    advice_from_card,
    diagnose,
    general_advice,
    load_cards,
    make_client,
    prepare_image,
    read_setting,
)

st.set_page_config(page_title="AgriLens", page_icon="🌾", layout="centered")


# ------------------------------------------------------------------ setup
def get_setting(name, default=None):
    """Streamlit Cloud keeps secrets in st.secrets; locally we also allow env vars."""
    try:
        return st.secrets[name]
    except Exception:  # noqa: BLE001
        return read_setting(name, default)


api_key = get_setting("GEMINI_API_KEY")
if not api_key:
    st.error(
        "Missing GEMINI_API_KEY. Add it to .streamlit/secrets.toml (local) "
        "or to the app's Secrets (Streamlit Cloud)."
    )
    st.stop()

MODEL = get_setting("GEMINI_MODEL", DEFAULT_MODEL)


@st.cache_resource
def get_client(key):
    return make_client(key)


client = get_client(api_key)
cards = load_cards()


# Caching saves free-tier quota: the same photo or the same card never calls the AI twice.
@st.cache_data(show_spinner=False)
def cached_diagnosis(image_bytes, model_name, _client):
    return diagnose(_client, model_name, image_bytes, "image/jpeg")


@st.cache_data(show_spinner=False)
def cached_card_advice(disease_name, card_body, language, model_name, _client):
    return advice_from_card(_client, model_name, disease_name, card_body, language)


@st.cache_data(show_spinner=False)
def cached_general_advice(image_bytes, language, model_name, _client):
    return general_advice(_client, model_name, image_bytes, "image/jpeg", language)


def friendly_error(error):
    message = str(error)
    if "429" in message or "RESOURCE_EXHAUSTED" in message:
        return "The free AI limit has been reached for now. Please wait a minute and try again."
    if "API_KEY" in message or "API key" in message or "403" in message:
        return "The API key was rejected. Check GEMINI_API_KEY in your secrets."
    if "404" in message or "not found" in message.lower():
        return "The AI model name was not found. Run check_setup.py and update GEMINI_MODEL."
    return f"Something went wrong: {message}"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Advice language", LANGUAGES)
    st.markdown("**Supported for now:** rice")
    st.caption("Can recognise: " + ", ".join(name.replace("Rice ", "") for name in DISEASE_NAMES.values()))
    st.caption("Other crops and unclear photos get cautious, unverified guidance.")


# -------------------------------------------------------------- main page
st.title("🌾 AgriLens")
st.caption("Take a photo of a rice plant problem and get practical, sourced advice in your language.")


def show_details(diagnosis):
    with st.expander("What the AI saw"):
        st.write(f"**Seen:** {diagnosis['visible_symptoms'] or 'n/a'}")
        st.write(f"**Why:** {diagnosis['reasoning'] or 'n/a'}")
        st.write(f"**Confidence:** {diagnosis['confidence']}  |  **Photo quality:** {diagnosis['photo_quality']}")


def show_source(card):
    if card.source_url:
        st.caption(f"Source: [{card.source_name}]({card.source_url})")
    elif card.source_name:
        st.caption(f"Source: {card.source_name}")


def show_result(diagnosis, image_bytes):
    label = diagnosis["disease"]
    confidence = diagnosis["confidence"]

    # Case 1: not a rice plant
    if not diagnosis["is_rice_plant"]:
        st.warning(
            "This does not look like a rice plant. AgriLens supports rice only for now. "
            "Please upload a clear photo of a rice leaf or grain head."
        )
        return

    # Case 2: bad photo
    if diagnosis["photo_quality"] == "poor":
        st.warning(
            "The photo is not clear enough. Take a closer photo of one affected leaf in good "
            "daylight, hold the phone steady, and avoid shadows."
        )
        show_details(diagnosis)
        return

    card = cards.get(label)

    # Case 3: not confident, so give cautious general guidance (Tier 3)
    if label == "unsure" or confidence == "low" or card is None:
        if label in DISEASE_NAMES:
            st.write(f"Possible match (low confidence): **{DISEASE_NAMES[label]}**")
        st.warning(
            "AgriLens is not sure about this photo. The guidance below is general and "
            "NOT verified. Please confirm with an agricultural officer."
        )
        with st.spinner("Preparing general guidance..."):
            try:
                text = cached_general_advice(image_bytes, language, MODEL, client)
            except Exception as error:  # noqa: BLE001
                st.error(friendly_error(error))
                return
        st.markdown(text)
        show_details(diagnosis)
        st.info(SAFETY_NOTE)
        return

    # Case 4: confident match with a card (Tier 1)
    st.subheader(card.name)
    if card.verified:
        st.success("Verified guidance, written from the source listed below.")
    else:
        st.info("Guidance based on our reference notes (source check in progress).")

    with st.spinner("Writing advice..."):
        try:
            text = cached_card_advice(card.name, card.body, language, MODEL, client)
        except Exception as error:  # noqa: BLE001
            st.error(friendly_error(error))
            return
    st.markdown(text)
    show_source(card)
    show_details(diagnosis)
    st.info(SAFETY_NOTE)


photo = st.file_uploader(
    "Upload or take a photo of the affected rice leaf or grain head",
    type=["jpg", "jpeg", "png", "webp"],
)

if not photo:
    st.info(
        "Tips for a good photo: one affected leaf or grain head, close up, in daylight, "
        "in focus, with the problem clearly visible."
    )
    st.stop()

try:
    image_bytes, _ = prepare_image(photo.getvalue())
except Exception:  # noqa: BLE001
    st.error("Sorry, this image could not be read. Please try a JPG or PNG photo.")
    st.stop()

st.image(image_bytes, caption="Your photo", width=360)
photo_id = hashlib.md5(image_bytes).hexdigest()

if st.button("Analyze photo", type="primary"):
    with st.spinner("Looking at your photo..."):
        try:
            diagnosis = cached_diagnosis(image_bytes, MODEL, client)
        except Exception as error:  # noqa: BLE001
            st.error(friendly_error(error))
            st.stop()
    st.session_state["result"] = {"photo_id": photo_id, "diagnosis": diagnosis}

result = st.session_state.get("result")
if result and result["photo_id"] == photo_id:
    show_result(result["diagnosis"], image_bytes)
