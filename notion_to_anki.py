#!/usr/bin/env python3
"""
Generate flashcards from a Notion page and upload them to Anki via AnkiConnect.

Required environment variables:
  - NOTION_TOKEN
  - OPENAI_API_KEY
  - NOTION_PAGE_ID (or pass --page-id)

Optional environment variables:
  - OPENAI_MODEL (default: gpt-4.1-mini)
  - ANKI_CONNECT_URL (default: http://127.0.0.1:8765)
  - ANKI_DECK (default: Notion Imports)
  - ANKI_NOTE_MODEL (default: Basic)
  - OPENAI_TEMPERATURE (default: 0.2)
  - CARDS_PER_CHUNK (default: 4)
  - MAX_CHARS_PER_CHUNK (default: 3500)
  - ALLOW_DUPLICATE_NOTES (default: false)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

import requests
from openai import OpenAI

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


@dataclass
class Section:
    title: str
    level: int
    body: str


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    normalized = normalized.strip("_")
    return normalized or "notion"


def notion_rich_text_to_plain(rich_text: list[dict[str, Any]] | None) -> str:
    if not rich_text:
        return ""
    return "".join(item.get("plain_text", "") for item in rich_text).strip()


def extract_page_title(page: dict[str, Any]) -> str:
    properties = page.get("properties", {})
    for prop in properties.values():
        if prop.get("type") == "title":
            title = notion_rich_text_to_plain(prop.get("title"))
            if title:
                return title
    return page.get("id", "Untitled Page")


class NotionClient:
    def __init__(self, token: str) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Notion-Version": NOTION_VERSION,
                "Content-Type": "application/json",
            }
        )

    def retrieve_page(self, page_id: str) -> dict[str, Any]:
        return self._request("GET", f"{NOTION_API_BASE}/pages/{page_id}")

    def list_block_children(self, block_id: str, start_cursor: str | None = None) -> dict[str, Any]:
        params = {}
        if start_cursor:
            params["start_cursor"] = start_cursor
        return self._request("GET", f"{NOTION_API_BASE}/blocks/{block_id}/children", params=params)

    def fetch_blocks_recursive(self, block_id: str, depth: int = 0) -> list[tuple[dict[str, Any], int]]:
        blocks: list[tuple[dict[str, Any], int]] = []
        cursor = None
        while True:
            payload = self.list_block_children(block_id, cursor)
            for block in payload.get("results", []):
                blocks.append((block, depth))
                if block.get("has_children"):
                    blocks.extend(self.fetch_blocks_recursive(block["id"], depth + 1))
            if not payload.get("has_more"):
                break
            cursor = payload.get("next_cursor")
        return blocks

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self.session.request(method, url, params=params, json=json_body, timeout=30)
        if response.status_code >= 400:
            raise RuntimeError(f"Notion API error ({response.status_code}): {response.text}")
        return response.json()


def block_to_markdown(block: dict[str, Any], depth: int) -> tuple[str, str | None]:
    block_type = block.get("type")
    data = block.get(block_type, {})
    indent = "  " * max(depth - 1, 0)

    if block_type in {"heading_1", "heading_2", "heading_3"}:
        text = notion_rich_text_to_plain(data.get("rich_text"))
        return ("heading", text)
    if block_type == "paragraph":
        text = notion_rich_text_to_plain(data.get("rich_text"))
        return ("content", text)
    if block_type == "bulleted_list_item":
        text = notion_rich_text_to_plain(data.get("rich_text"))
        return ("content", f"{indent}- {text}" if text else None)
    if block_type == "numbered_list_item":
        text = notion_rich_text_to_plain(data.get("rich_text"))
        return ("content", f"{indent}1. {text}" if text else None)
    if block_type == "to_do":
        text = notion_rich_text_to_plain(data.get("rich_text"))
        checked = data.get("checked", False)
        box = "[x]" if checked else "[ ]"
        return ("content", f"{indent}- {box} {text}" if text else None)
    if block_type == "quote":
        text = notion_rich_text_to_plain(data.get("rich_text"))
        return ("content", f"> {text}" if text else None)
    if block_type == "callout":
        text = notion_rich_text_to_plain(data.get("rich_text"))
        return ("content", f"Note: {text}" if text else None)
    if block_type == "code":
        text = notion_rich_text_to_plain(data.get("rich_text"))
        language = data.get("language", "text")
        if not text:
            return ("content", None)
        code_block = f"```{language}\n{text}\n```"
        return ("content", code_block)
    if block_type == "toggle":
        text = notion_rich_text_to_plain(data.get("rich_text"))
        return ("content", f"Toggle: {text}" if text else None)
    if block_type == "child_page":
        title = data.get("title", "").strip()
        return ("content", f"Child page: {title}" if title else None)

    return ("content", None)


def blocks_to_sections(blocks_with_depth: list[tuple[dict[str, Any], int]]) -> list[Section]:
    sections: list[Section] = []
    current_title = "Introduction"
    current_level = 0
    current_lines: list[str] = []

    def flush_current() -> None:
        nonlocal current_lines
        body = "\n".join(line for line in current_lines if line and line.strip()).strip()
        if body:
            sections.append(Section(title=current_title, level=current_level, body=body))
        current_lines = []

    for block, depth in blocks_with_depth:
        kind, text = block_to_markdown(block, depth)
        if kind == "heading":
            heading_text = (text or "").strip()
            if not heading_text:
                continue
            flush_current()
            current_title = heading_text
            block_type = block.get("type")
            current_level = 1 if block_type == "heading_1" else 2 if block_type == "heading_2" else 3
            continue
        if text:
            current_lines.append(text)

    flush_current()
    return sections


def split_sections(sections: list[Section], max_chars: int) -> list[Section]:
    if max_chars <= 0:
        return sections

    output: list[Section] = []
    for section in sections:
        if len(section.body) <= max_chars:
            output.append(section)
            continue

        paragraphs = [p.strip() for p in section.body.split("\n\n") if p.strip()]
        part_lines: list[str] = []
        part_size = 0
        part_index = 1

        for paragraph in paragraphs:
            p_size = len(paragraph) + 2
            if part_lines and (part_size + p_size) > max_chars:
                output.append(
                    Section(
                        title=f"{section.title} (part {part_index})",
                        level=section.level,
                        body="\n\n".join(part_lines),
                    )
                )
                part_index += 1
                part_lines = [paragraph]
                part_size = p_size
            else:
                part_lines.append(paragraph)
                part_size += p_size

        if part_lines:
            output.append(
                Section(
                    title=f"{section.title} (part {part_index})" if part_index > 1 else section.title,
                    level=section.level,
                    body="\n\n".join(part_lines),
                )
            )

    return output


def parse_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse model JSON output: {exc}") from exc


def generate_cards_for_section(
    client: OpenAI,
    model: str,
    page_title: str,
    section: Section,
    cards_per_chunk: int,
    temperature: float,
) -> list[dict[str, Any]]:
    system_prompt = (
        "You create high-quality study flashcards from notes. "
        "Return strict JSON with key 'cards' only. "
        "Each card must include: front (question), back (concise answer), tags (array of short strings). "
        "Avoid trivial cards. Prefer conceptual understanding and recall."
    )
    user_prompt = (
        f"Page title: {page_title}\n"
        f"Section title: {section.title}\n"
        f"Section content:\n{section.body}\n\n"
        f"Create up to {cards_per_chunk} useful flashcards."
    )

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content or "{}"
    parsed = parse_json_object(content)
    cards = parsed.get("cards", [])
    if not isinstance(cards, list):
        return []

    validated: list[dict[str, Any]] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        front = str(card.get("front", "")).strip()
        back = str(card.get("back", "")).strip()
        tags_raw = card.get("tags", [])
        tags: list[str] = []
        if isinstance(tags_raw, list):
            tags = [slugify(str(tag)) for tag in tags_raw if str(tag).strip()]

        if front and back:
            validated.append({"front": front, "back": back, "tags": tags})

    return validated


class AnkiConnectClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def call(self, action: str, params: dict[str, Any] | None = None) -> Any:
        payload = {
            "action": action,
            "version": 6,
            "params": params or {},
        }
        response = requests.post(self.endpoint, json=payload, timeout=20)
        if response.status_code >= 400:
            raise RuntimeError(f"AnkiConnect error ({response.status_code}): {response.text}")
        data = response.json()
        if data.get("error"):
            raise RuntimeError(f"AnkiConnect action '{action}' failed: {data['error']}")
        return data.get("result")

    def ensure_deck(self, deck_name: str) -> None:
        self.call("createDeck", {"deck": deck_name})

    def model_exists(self, model_name: str) -> bool:
        model_names = self.call("modelNames") or []
        return model_name in model_names

    def add_notes(self, notes: list[dict[str, Any]]) -> list[Any]:
        return self.call("addNotes", {"notes": notes})


def deduplicate_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    output: list[dict[str, Any]] = []
    for card in cards:
        key = (card["front"].strip().lower(), card["back"].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        output.append(card)
    return output


def build_anki_notes(
    cards: list[dict[str, Any]],
    deck_name: str,
    model_name: str,
    base_tags: list[str],
    allow_duplicates: bool,
) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for card in cards:
        tags = sorted(set(base_tags + card.get("tags", [])))
        notes.append(
            {
                "deckName": deck_name,
                "modelName": model_name,
                "fields": {
                    "Front": card["front"],
                    "Back": card["back"],
                },
                "tags": tags,
                "options": {
                    "allowDuplicate": allow_duplicates,
                },
            }
        )
    return notes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Anki flashcards from a Notion page.")
    parser.add_argument("--page-id", default=get_env("NOTION_PAGE_ID"))
    parser.add_argument("--deck", default=get_env("ANKI_DECK", "Notion Imports"))
    parser.add_argument("--model-name", default=get_env("ANKI_NOTE_MODEL", "Basic"))
    parser.add_argument("--openai-model", default=get_env("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument(
        "--cards-per-chunk",
        type=int,
        default=int(get_env("CARDS_PER_CHUNK", "4")),
    )
    parser.add_argument(
        "--max-chars-per-chunk",
        type=int,
        default=int(get_env("MAX_CHARS_PER_CHUNK", "3500")),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(get_env("OPENAI_TEMPERATURE", "0.2")),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    notion_token = get_env("NOTION_TOKEN")
    openai_key = get_env("OPENAI_API_KEY")
    anki_url = get_env("ANKI_CONNECT_URL", "http://127.0.0.1:8765")
    allow_duplicates = parse_bool(get_env("ALLOW_DUPLICATE_NOTES", "false"), default=False)

    missing: list[str] = []
    if not notion_token:
        missing.append("NOTION_TOKEN")
    if not openai_key:
        missing.append("OPENAI_API_KEY")
    if not args.page_id:
        missing.append("NOTION_PAGE_ID (or pass --page-id)")
    if missing:
        print("Missing required configuration:", ", ".join(missing), file=sys.stderr)
        return 1

    notion = NotionClient(notion_token)
    openai_client = OpenAI(api_key=openai_key)
    anki = AnkiConnectClient(anki_url)

    print(f"Fetching Notion page {args.page_id}...")
    page = notion.retrieve_page(args.page_id)
    page_title = extract_page_title(page)
    blocks = notion.fetch_blocks_recursive(args.page_id)
    if not blocks:
        print("No block content found in this Notion page.", file=sys.stderr)
        return 1

    sections = blocks_to_sections(blocks)
    sections = split_sections(sections, max_chars=args.max_chars_per_chunk)
    if not sections:
        print("No usable section content found after parsing the page.", file=sys.stderr)
        return 1

    all_cards: list[dict[str, Any]] = []
    for index, section in enumerate(sections, start=1):
        print(f"Generating cards for section {index}/{len(sections)}: {section.title}")
        cards = generate_cards_for_section(
            client=openai_client,
            model=args.openai_model,
            page_title=page_title,
            section=section,
            cards_per_chunk=args.cards_per_chunk,
            temperature=args.temperature,
        )
        all_cards.extend(cards)

    all_cards = deduplicate_cards(all_cards)
    if not all_cards:
        print("No valid cards were generated.", file=sys.stderr)
        return 1

    base_tags = [slugify(page_title), "notion_import"]
    notes = build_anki_notes(
        cards=all_cards,
        deck_name=args.deck,
        model_name=args.model_name,
        base_tags=base_tags,
        allow_duplicates=allow_duplicates,
    )

    print(f"Generated {len(all_cards)} unique cards from '{page_title}'.")
    if args.dry_run:
        preview = [{"front": c["front"], "back": c["back"], "tags": c["tags"]} for c in all_cards[:5]]
        print("Dry run enabled. First cards preview:")
        print(json.dumps(preview, indent=2, ensure_ascii=True))
        return 0

    anki.ensure_deck(args.deck)
    if not anki.model_exists(args.model_name):
        print(
            f"Anki note model '{args.model_name}' does not exist. "
            "Use an existing model (for example 'Basic') via --model-name or ANKI_NOTE_MODEL.",
            file=sys.stderr,
        )
        return 1

    print(f"Uploading {len(notes)} notes to Anki deck '{args.deck}'...")
    result = anki.add_notes(notes)
    inserted = sum(1 for item in result if isinstance(item, int))
    rejected = len(result) - inserted
    print(f"Upload complete: {inserted} inserted, {rejected} rejected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
