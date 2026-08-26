"""Lets you run the package as `python -m pdf_chat ...`."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
