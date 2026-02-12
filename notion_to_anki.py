#!/usr/bin/env python3
"""
Compatibility wrapper.

Prefer:
  python anki_flashcard_generator.py ...
"""

from anki_flashcard_generator import main


if __name__ == "__main__":
    raise SystemExit(main())
