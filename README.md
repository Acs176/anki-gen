# Notion to Anki Flashcard Generator

Python script to:
1. Read content from a Notion page
2. Chunk content by headings
3. Generate flashcards with OpenAI
4. Upload notes to Anki via AnkiConnect

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Required Environment Variables

```bash
export NOTION_TOKEN="secret_xxx"
export OPENAI_API_KEY="sk-..."
export NOTION_PAGE_ID="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## Optional Environment Variables

```bash
export OPENAI_MODEL="gpt-4.1-mini"
export ANKI_CONNECT_URL="http://127.0.0.1:8765"
export ANKI_DECK="Notion Imports"
export ANKI_NOTE_MODEL="Basic"
export OPENAI_TEMPERATURE="0.2"
export CARDS_PER_CHUNK="4"
export MAX_CHARS_PER_CHUNK="3500"
export ALLOW_DUPLICATE_NOTES="false"
```

## Run

Dry run (no Anki upload):

```bash
python notion_to_anki.py --dry-run
```

Upload to Anki:

```bash
python notion_to_anki.py
```

Override settings at runtime:

```bash
python notion_to_anki.py \
  --page-id "<NOTION_PAGE_ID>" \
  --deck "My Deck" \
  --model-name "Basic" \
  --openai-model "gpt-4.1-mini" \
  --cards-per-chunk 5 \
  --max-chars-per-chunk 3000 \
  --temperature 0.2
```

## Notes

- The script expects Anki to be running with the AnkiConnect add-on.
- The selected note model must include `Front` and `Back` fields.
- Card quality depends on your Notion content quality and structure.
