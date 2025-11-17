import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


try:
    from llama_index.readers.docling import DoclingReader
except ImportError:  # pragma: no cover - dependency is optional at runtime
    DoclingReader = None


_HEADING_PATTERN = re.compile(r"^(?P<heading>#{1,6}\s+.+)$", re.MULTILINE)


class IngestionHelper:
    """Helper class to transform a file into a list of documents.

    This class should be used to transform a file into a list of documents.
    These methods are thread-safe (and multiprocessing-safe).
    """

    @staticmethod
    def transform_file_into_documents(
        file_name: str, file_data: Path
    ) -> list[Document]:
        documents = IngestionHelper._load_file_to_documents(file_name, file_data)
        for document in documents:
            document.metadata["file_name"] = file_name
        IngestionHelper._exclude_metadata(documents)
        return documents

    @staticmethod
    def _load_file_to_documents(file_name: str, file_data: Path) -> list[Document]:
        logger.debug("Transforming file_name=%s into documents", file_name)
        if DoclingReader is None:
            raise ImportError(
                "DoclingReader is required but not available. Install the "
                "`llama-index-readers-docling` extra to enable document ingestion."
            )

        reader = DoclingReader()
        docling_documents = reader.load_data(file_data)

        documents: list[Document] = []
        for source_document in docling_documents:
            documents.extend(
                IngestionHelper._chunk_document_by_markdown_heading(source_document)
            )

        for document in documents:
            document.text = document.text.replace("\u0000", "")

        return documents

    @staticmethod
    def _chunk_document_by_markdown_heading(source_document: Document) -> list[Document]:
        text = source_document.text or ""
        chunks: list[Document] = []

        heading_matches = list(_HEADING_PATTERN.finditer(text))
        if not heading_matches:
            cloned_document = Document(text=text, doc_id=source_document.doc_id)
            cloned_document.metadata = IngestionHelper._clone_metadata(source_document)
            chunks.append(cloned_document)
            return chunks

        # Capture text that appears before the first heading as its own chunk
        first_heading_start = heading_matches[0].start()
        prefix = text[:first_heading_start].strip()
        if prefix:
            chunk_doc = Document(
                text=prefix,
                doc_id=f"{source_document.doc_id}_preamble",
            )
            chunk_doc.metadata = IngestionHelper._clone_metadata(source_document)
            chunk_doc.metadata["chapter_title"] = "Preamble"
            chunk_doc.metadata["chapter_level"] = 0
            chunks.append(chunk_doc)

        for idx, match in enumerate(heading_matches):
            start = match.start()
            end = (
                heading_matches[idx + 1].start()
                if idx + 1 < len(heading_matches)
                else len(text)
            )
            chunk_text = text[start:end].strip()
            if not chunk_text:
                continue
            heading_line = match.group("heading")
            chunk_doc = Document(
                text=chunk_text,
                doc_id=f"{source_document.doc_id}_chapter_{idx}",
            )
            chunk_doc.metadata = IngestionHelper._clone_metadata(source_document)
            chunk_doc.metadata["chapter_title"] = heading_line.lstrip("#").strip()
            chunk_doc.metadata["chapter_level"] = heading_line.count("#")
            chunks.append(chunk_doc)

        return chunks

    @staticmethod
    def _clone_metadata(document: Document) -> dict[str, Any]:
        metadata = getattr(document, "metadata", {}) or {}
        return deepcopy(metadata)

    @staticmethod
def _exclude_metadata(documents: list[Document]) -> None:
        logger.debug("Excluding metadata from count=%s documents", len(documents))
        for document in documents:
            document.metadata["doc_id"] = document.doc_id
            # We don't want the Embeddings search to receive this metadata
            document.excluded_embed_metadata_keys = ["doc_id"]
            # We don't want the LLM to receive these metadata in the context
            document.excluded_llm_metadata_keys = ["file_name", "doc_id", "page_label"]
