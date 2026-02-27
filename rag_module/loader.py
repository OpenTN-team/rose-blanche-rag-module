"""
loader.py
Loads and chunks documents from various formats (PDF, DOCX, TXT, CSV, JSON).
Each chunk becomes a Fragment ready for indexing.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

from .indexer import Fragment

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads documents and splits them into overlapping text fragments (chunks).

    Supported formats: .txt, .pdf, .docx, .csv, .json, .md

    Args:
        chunk_size:    Approximate character length of each fragment.
        chunk_overlap: Number of characters to overlap between consecutive chunks.
                       Helps preserve context at chunk boundaries.
        min_chunk_len: Chunks shorter than this are discarded (noise filter).
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_len: int = 40,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_len = min_chunk_len

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_file(self, path: str) -> List[Fragment]:
        """Load a single file and return its fragments."""
        path_obj = Path(path)
        ext = path_obj.suffix.lower()
        loaders = {
            ".txt": self._load_txt,
            ".md":  self._load_txt,
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".csv": self._load_csv,
            ".json": self._load_json,
        }
        if ext not in loaders:
            logger.warning("Unsupported file format: %s — skipped.", path)
            return []

        raw_pages = loaders[ext](path)
        fragments: List[Fragment] = []
        for page_num, text in raw_pages:
            chunks = self._split_text(text)
            for chunk in chunks:
                fragments.append(
                    Fragment(
                        id=0,  # will be set by VectorIndex.add()
                        text=chunk,
                        source=path_obj.name,
                        page=page_num,
                    )
                )
        logger.info("Loaded %d fragments from %s.", len(fragments), path_obj.name)
        return fragments

    def load_directory(self, directory: str, recursive: bool = True) -> List[Fragment]:
        """Load all supported files in a directory."""
        all_fragments: List[Fragment] = []
        pattern = "**/*" if recursive else "*"
        for p in Path(directory).glob(pattern):
            if p.is_file():
                all_fragments.extend(self.load_file(str(p)))
        logger.info(
            "Directory load complete: %d total fragments from %s.",
            len(all_fragments),
            directory,
        )
        return all_fragments

    def load_texts(
        self, texts: List[str], source: str = "inline"
    ) -> List[Fragment]:
        """Directly load a list of raw text strings (no file I/O)."""
        fragments: List[Fragment] = []
        for i, text in enumerate(texts):
            for chunk in self._split_text(text):
                fragments.append(Fragment(id=0, text=chunk, source=source, page=i + 1))
        return fragments

    # ------------------------------------------------------------------
    # Format loaders  (return list of (page_number | None, raw_text))
    # ------------------------------------------------------------------

    def _load_txt(self, path: str):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return [(None, f.read())]

    def _load_pdf(self, path: str):
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required for PDF support. "
                "Install it with: pip install pdfplumber"
            )
        pages = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((i, text))
        return pages

    def _load_docx(self, path: str):
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "python-docx is required for DOCX support. "
                "Install it with: pip install python-docx"
            )
        doc = Document(path)
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [(None, full_text)]

    def _load_csv(self, path: str):
        """Each row becomes a single text fragment (columns joined)."""
        rows = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=1):
                text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
                rows.append((i, text))
        return rows

    def _load_json(self, path: str):
        """Handles JSON array of strings or objects."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [
                (i + 1, json.dumps(item, ensure_ascii=False))
                for i, item in enumerate(data)
            ]
        return [(None, json.dumps(data, ensure_ascii=False))]

    # ------------------------------------------------------------------
    # Text chunking
    # ------------------------------------------------------------------

    def _split_text(self, text: str) -> List[str]:
        """
        Split *text* into overlapping chunks of approximately `chunk_size`
        characters, respecting sentence/paragraph boundaries where possible.
        """
        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        chunks: List[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + self.chunk_size, length)

            # Prefer to cut at the last sentence-end before `end`,
            # but only if the boundary leaves a chunk large enough to be meaningful
            # (avoids start going backwards when the boundary is too close to start).
            if end < length:
                boundary = self._find_sentence_boundary(text, start, end)
                if boundary > start + self.chunk_overlap:
                    end = boundary

            chunk = text[start:end].strip()
            if len(chunk) >= self.min_chunk_len:
                chunks.append(chunk)

            if end >= length:
                break

            # Advance with overlap — guarantee forward progress to avoid infinite loops
            next_start = end - self.chunk_overlap
            start = max(next_start, start + 1)

        return chunks

    @staticmethod
    def _find_sentence_boundary(text: str, start: int, end: int) -> int:
        """Find the last sentence-ending punctuation within (start, end]."""
        for punct in (".", "!", "?", "\n"):
            pos = text.rfind(punct, start, end)
            if pos != -1:
                return pos + 1
        return end
