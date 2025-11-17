from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from private_gpt.server.chunks.chunks_service import Chunk


@dataclass(frozen=True)
class CitationDetail:
    """Description for a single inline citation."""

    index: int
    label: str
    description: str
    chunk: Chunk


class InlineCitationFormatter:
    """Utility responsible for rendering inline citation markers.

    This follows the guidance from the LlamaIndex citation workflow example and adds
    numeric inline markers plus a human-readable source list that can later be hooked
    into a dedicated LlamaIndex Workflow/Agent pipeline.
    """

    def __init__(self, *, heading: str = "Sources") -> None:
        self.heading = heading

    def _build_citation_details(self, sources: Iterable[Chunk]) -> list[CitationDetail]:
        unique_chunks: list[Chunk] = []
        seen_chunks: set[tuple[str, str]] = set()
        for chunk in sources:
            key = (chunk.document.doc_id, chunk.text)
            if key in seen_chunks:
                continue
            seen_chunks.add(key)
            unique_chunks.append(chunk)

        details: list[CitationDetail] = []
        for index, chunk in enumerate(unique_chunks, start=1):
            metadata = chunk.document.doc_metadata or {}
            file_name = metadata.get("file_name")
            page_label = metadata.get("page_label")
            doc_reference = file_name or chunk.document.doc_id
            page_suffix = f" (page {page_label})" if page_label else ""
            description = f"{doc_reference}{page_suffix}".strip()
            details.append(
                CitationDetail(
                    index=index,
                    label=f"[{index}]",
                    description=description,
                    chunk=chunk,
                )
            )

        return details

    def build_suffix(self, sources: Iterable[Chunk]) -> str:
        """Return a suffix that contains inline markers and a source section."""

        details = self._build_citation_details(sources)
        if not details:
            return ""

        inline_markers = " ".join(detail.label for detail in details)
        sources_lines = [f"{detail.label} {detail.description}" for detail in details]
        sources_block = f"{self.heading}\n" + "\n".join(sources_lines)
        return f" {inline_markers}\n\n{sources_block}"

    def decorate(self, text: str, sources: Iterable[Chunk]) -> str:
        """Decorate a completed answer with inline citations if available."""

        suffix = self.build_suffix(sources)
        if not suffix:
            return text
        return f"{text.rstrip()}{suffix}"
