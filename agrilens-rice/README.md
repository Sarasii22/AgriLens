# AgriLens (rice edition)

AgriLens helps small-scale rice farmers in Sri Lanka understand a problem in their crop.
Take a photo of a rice leaf or grain head and get simple, sourced advice in English, Sinhala or Tamil.

Built for the ML Empowerment Build Challenge.

## How it works

1. **Look:** an AI vision model (Google Gemini) examines the photo and must choose ONE label from a fixed list:
   leaf blast, neck blast, brown spot, narrow brown spot, hispa, healthy, or "unsure".
2. **Retrieve:** the app loads the matching reference card from `cards/`. These cards are short notes
   written from Department of Agriculture Sri Lanka pages, each with its source link.
3. **Write:** the AI writes advice using ONLY that card, in the farmer's language, with no pesticide names or doses.
4. **Be honest:** if the photo is unclear, not rice, or the AI is unsure, the app says so and gives cautious,
   clearly labelled "unverified" general guidance instead of guessing.

## Run it on your computer

    python -m venv .venv
    # Windows PowerShell:  .venv\Scripts\Activate.ps1      Mac/Linux:  source .venv/bin/activate
    pip install -r requirements.txt
    # copy .streamlit/secrets.toml.example to .streamlit/secrets.toml and paste your Gemini API key
    python check_setup.py
    streamlit run app.py

## Test accuracy

Put your own test photos in `test_images/<label>/` (labels: leaf_blast, neck_blast, brown_spot,
narrow_brown_spot, hispa, healthy, not_rice), then run `python evaluate.py`.

## Project files

- `app.py` - the Streamlit web app
- `engine.py` - diagnosis, card lookup and advice logic
- `cards/` - the knowledge base (one Markdown file per condition)
- `evaluate.py` - measures accuracy on your test photos
- `check_setup.py` - checks your API key and model

## Limits (please read)

- Rice only for now. Other crops are a future extension.
- AgriLens gives general guidance, not a certain diagnosis. Farmers should confirm serious problems
  with a Department of Agriculture or agrarian services officer.
- Reference cards are short summaries written in our own words; see each card for its source.

## Sources

- Department of Agriculture Sri Lanka, Rice Research and Development Institute, "Rice In Sri Lanka - Rice Diseases" pages.
- Add every other source you use here.
