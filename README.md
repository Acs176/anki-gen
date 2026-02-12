# Anki Flashcard Generator (LLM)

Python script to:
1. Read content from multiple sources (`stdin`, raw text, file, or Notion)
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
export OPENAI_API_KEY="sk-..."
```

## Optional Environment Variables

```bash
export FLASHCARD_SOURCE="stdin" # stdin | text | file | notion
export FLASHCARD_TITLE="My Notes"
export FLASHCARD_TEXT="# Topic\nSome text"
export FLASHCARD_FILE="./notes.md"

# Notion mode only:
export NOTION_TOKEN="secret_xxx"
export NOTION_PAGE_ID="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

export OPENAI_MODEL="gpt-4.1-mini"
export ANKI_CONNECT_URL="http://127.0.0.1:8765"
export ANKI_DECK="LLM Flashcards"
export ANKI_NOTE_MODEL="Basic"
export OPENAI_TEMPERATURE="0.2"
export CARDS_PER_CHUNK="4"
export MAX_CHARS_PER_CHUNK="3500"
export ALLOW_DUPLICATE_NOTES="false"
```

## Run

Dry run (no Anki upload):

```bash
cat notes.md | python anki_flashcard_generator.py --source stdin --dry-run
```

Upload to Anki:

```bash
cat notes.md | python anki_flashcard_generator.py --source stdin
```

Use raw text:

```bash
python anki_flashcard_generator.py \
  --source text \
  --title "Biology Basics" \
  --text "# Cell Theory\nCells are the basic unit of life." \
  --dry-run
```

Use a markdown file:

```bash
python anki_flashcard_generator.py \
  --source file \
  --file ./notes.md \
  --title "My File Notes"
```

Use Notion (optional integration):

```bash
python anki_flashcard_generator.py \
  --source notion \
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
- For Notion mode, share the page with your Notion integration/token.
- Card quality depends on content quality and structure.
- `notion_to_anki.py` remains as a compatibility wrapper around `anki_flashcard_generator.py`.
