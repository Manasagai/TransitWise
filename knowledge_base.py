"""Local document ingestion and provenance for the passenger-advisor knowledge base."""

from __future__ import annotations

from datetime import date, datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse


MAX_DOCUMENT_BYTES = 15 * 1024 * 1024
MAX_PASSAGE_CHARACTERS = 1200
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".html", ".htm", ".md", ".markdown"}
OFFICIAL_TGSRTC_HOST = "tgsrtc.telangana.gov.in"
MANIFEST_FILENAME = "manifest.json"


class KnowledgeBaseError(ValueError):
    """An uploaded knowledge source cannot safely be stored or indexed."""


class _VisibleHtmlText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._hidden_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: Sequence[Tuple[str, Optional[str]]],
    ) -> None:
        if tag.lower() in {"script", "style", "noscript", "header", "footer", "nav", "aside"}:
            self._hidden_depth += 1
        elif tag.lower() in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")
        elif tag.lower() == "img":
            alt_text = next(
                (value for name, value in attrs if name.lower() == "alt" and value),
                None,
            )
            if alt_text:
                self.parts.append(f" {alt_text} ")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "header", "footer", "nav", "aside"} and self._hidden_depth:
            self._hidden_depth -= 1
        elif tag.lower() in {"p", "div", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.parts.append(data)


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    paragraphs = [line for line in lines if line]
    return "\n\n".join(paragraphs).strip()


def extract_document_text(content: bytes, filename: str) -> str:
    """Extract text from a supported local file without executing embedded content."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise KnowledgeBaseError(
            "Unsupported document type. Upload a PDF, TXT, HTML, or Markdown file."
        )
    if not content:
        raise KnowledgeBaseError("The uploaded document is empty.")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise KnowledgeBaseError("The uploaded document exceeds the 15 MB limit.")

    if extension in {".txt", ".md", ".markdown"}:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise KnowledgeBaseError(
                "Text documents must be encoded as UTF-8."
            ) from error
    elif extension in {".html", ".htm"}:
        try:
            parser = _VisibleHtmlText()
            parser.feed(content.decode("utf-8-sig"))
            parser.close()
        except (UnicodeDecodeError, ValueError) as error:
            raise KnowledgeBaseError("The HTML document could not be parsed.") from error
        text = " ".join(parser.parts)
    else:
        try:
            from pypdf import PdfReader
            from pypdf.errors import PdfReadError
            from io import BytesIO

            reader = PdfReader(BytesIO(content), strict=True)
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError as error:
            raise KnowledgeBaseError(
                "PDF ingestion requires the declared pypdf dependency."
            ) from error
        except (PdfReadError, OSError, ValueError) as error:
            raise KnowledgeBaseError(
                f"The PDF document could not be read: {error}"
            ) from error

    normalized = _normalize_text(text)
    if not normalized:
        raise KnowledgeBaseError(
            "No readable text was found. Scanned image-only PDFs are not supported."
        )
    return normalized


def split_passages(text: str, max_characters: int = MAX_PASSAGE_CHARACTERS) -> List[str]:
    if max_characters < 1:
        raise ValueError("max_characters must be positive")
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    passages: List[str] = []
    for paragraph in paragraphs:
        remaining = paragraph
        while len(remaining) > max_characters:
            split_at = remaining.rfind(" ", 0, max_characters + 1)
            if split_at <= 0:
                split_at = max_characters
            passage = remaining[:split_at].strip()
            if passage:
                passages.append(passage)
            remaining = remaining[split_at:].strip()
        if remaining:
            passages.append(remaining)
    return passages


def _read_manifest(knowledge_dir: Path) -> Dict[str, Any]:
    path = knowledge_dir / MANIFEST_FILENAME
    if not path.exists():
        return {"schema_version": 1, "sources": []}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KnowledgeBaseError(f"The knowledge-source manifest is invalid: {error}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or not isinstance(manifest.get("sources"), list)
    ):
        raise KnowledgeBaseError("The knowledge-source manifest has an unsupported schema.")
    return manifest


def _validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    required = ("source_title", "publisher", "document_type", "source_class")
    if any(not isinstance(metadata.get(key), str) or not metadata[key].strip() for key in required):
        raise KnowledgeBaseError(
            "Source title, publisher, document type, and source category are required."
        )
    source_class = metadata["source_class"]
    if source_class not in {
        "official_tgsrtc",
        "general_reference",
        "illustrative_legacy",
    }:
        raise KnowledgeBaseError("Select a supported source category.")

    source_url = str(metadata.get("source_url") or "").strip()
    if source_url:
        parsed_url = urlparse(source_url)
        if parsed_url.scheme != "https" or not parsed_url.hostname:
            raise KnowledgeBaseError("Source URLs must use HTTPS.")
    if source_class == "official_tgsrtc":
        parsed_url = urlparse(source_url)
        hostname = (parsed_url.hostname or "").lower().rstrip(".")
        if hostname != OFFICIAL_TGSRTC_HOST and not hostname.endswith(
            f".{OFFICIAL_TGSRTC_HOST}"
        ):
            raise KnowledgeBaseError(
                "An official TGSRTC source must use tgsrtc.telangana.gov.in."
            )
    for date_key in ("published_date", "updated_date"):
        value = metadata.get(date_key)
        if value in (None, ""):
            metadata[date_key] = None
            continue
        try:
            date.fromisoformat(str(value))
        except ValueError as error:
            raise KnowledgeBaseError(
                f"{date_key.replace('_', ' ').capitalize()} must use YYYY-MM-DD."
            ) from error
        metadata[date_key] = str(value)
    source_retrieved_at = metadata.get("retrieved_at")
    if source_retrieved_at in (None, ""):
        metadata["retrieved_at"] = None
    else:
        metadata["retrieved_at"] = _normalize_retrieval_time(str(source_retrieved_at))
    metadata["source_url"] = source_url
    return metadata


def _normalize_retrieval_time(value: str) -> str:
    try:
        parsed_retrieval = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise KnowledgeBaseError(
            "Source retrieval time must use an ISO-8601 timestamp with a timezone."
        ) from error
    if parsed_retrieval.tzinfo is None:
        raise KnowledgeBaseError(
            "Source retrieval time must include a timezone, preferably UTC."
        )
    return parsed_retrieval.astimezone(timezone.utc).isoformat(timespec="seconds")


def ingest_local_document(
    content: bytes,
    original_filename: str,
    metadata: Dict[str, Any],
    knowledge_dir: Path | str,
    reviewed: bool = False,
) -> Dict[str, Any]:
    """Persist one source file and its provenance manifest record."""
    knowledge_dir = Path(knowledge_dir)
    if not isinstance(content, bytes):
        raise KnowledgeBaseError("Document content must be bytes.")
    text = extract_document_text(content, original_filename)
    safe_metadata = _validate_metadata(dict(metadata))
    digest = sha256(content).hexdigest()
    text_digest = sha256(text.encode("utf-8")).hexdigest()
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(knowledge_dir)
    for source in manifest["sources"]:
        if source.get("sha256") == digest or source.get("content_sha256") == text_digest:
            raise KnowledgeBaseError(
                f"This exact document is already registered as "
                f"{source.get('original_filename', 'a local source')}."
            )

    safe_name = Path(original_filename).name
    extension = Path(safe_name).suffix.lower()
    if not safe_name or extension not in SUPPORTED_EXTENSIONS:
        raise KnowledgeBaseError("The original filename has an unsupported extension.")
    stored_filename = f"{digest}{extension}"
    originals_dir = knowledge_dir / "originals"
    originals_dir.mkdir(parents=True, exist_ok=True)
    stored_path = originals_dir / stored_filename
    if stored_path.exists():
        raise KnowledgeBaseError("A stored file already exists for this document hash.")

    ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    source_record = {
        **safe_metadata,
        "source_id": digest,
        "original_filename": safe_name,
        "stored_filename": stored_filename,
        "sha256": digest,
        "content_sha256": text_digest,
        "ingested_at": ingested_at,
        "document_type": safe_metadata["document_type"].strip(),
        "reviewed": bool(reviewed),
        "ingestion_status": "success",
        "passage_count": len(split_passages(text)),
    }
    if source_record["source_class"] == "illustrative_legacy":
        source_record["index_status"] = "excluded: illustrative content"
    elif not reviewed:
        source_record["index_status"] = "pending local source review"
    else:
        source_record["index_status"] = "indexed"

    stored_path.write_bytes(content)
    try:
        next_manifest = {
            "schema_version": 1,
            "sources": [*manifest["sources"], source_record],
        }
        temporary_manifest = knowledge_dir / f"{MANIFEST_FILENAME}.tmp"
        temporary_manifest.write_text(
            json.dumps(next_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(knowledge_dir / MANIFEST_FILENAME)
    except OSError as error:
        stored_path.unlink(missing_ok=True)
        raise KnowledgeBaseError(
            f"The source was extracted but could not be registered: {error}"
        ) from error
    return source_record


def update_source_retrieval_time(
    knowledge_dir: Path | str,
    source_id: str,
    retrieved_at: str,
) -> Dict[str, Any]:
    """Update a local source's known fetch time without altering its original."""
    knowledge_dir = Path(knowledge_dir)
    manifest = _read_manifest(knowledge_dir)
    matching_source = next(
        (
            source
            for source in manifest["sources"]
            if source.get("source_id") == source_id
        ),
        None,
    )
    if matching_source is None:
        raise KnowledgeBaseError("The selected local source no longer exists.")
    if "ingested_at" not in matching_source and matching_source.get("retrieved_at"):
        matching_source["ingested_at"] = matching_source["retrieved_at"]
        matching_source["retrieved_at"] = matching_source.get("source_retrieved_at")
    stored_path = (
        knowledge_dir
        / "originals"
        / Path(matching_source.get("stored_filename", "")).name
    )
    if not stored_path.is_file():
        raise KnowledgeBaseError("The original local source file is missing.")
    content = stored_path.read_bytes()
    if sha256(content).hexdigest() != matching_source.get("sha256"):
        raise KnowledgeBaseError(
            "The original local source hash does not match its registered provenance."
        )
    matching_source["passage_count"] = len(
        split_passages(
            extract_document_text(
                content,
                matching_source["original_filename"],
            )
        )
    )
    matching_source["retrieved_at"] = _normalize_retrieval_time(retrieved_at)
    temporary_manifest = knowledge_dir / f"{MANIFEST_FILENAME}.tmp"
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(knowledge_dir / MANIFEST_FILENAME)
    except OSError as error:
        temporary_manifest.unlink(missing_ok=True)
        raise KnowledgeBaseError(
            f"Source metadata could not be updated: {error}"
        ) from error
    return matching_source


def list_knowledge_sources(knowledge_dir: Path | str) -> List[Dict[str, Any]]:
    """Return persisted source records with current file-integrity/index status."""
    knowledge_dir = Path(knowledge_dir)
    manifest = _read_manifest(knowledge_dir)
    sources: List[Dict[str, Any]] = []
    for record in manifest["sources"]:
        source = dict(record)
        if "ingested_at" not in source and source.get("retrieved_at"):
            source["ingested_at"] = source["retrieved_at"]
            source["retrieved_at"] = source.get("source_retrieved_at")
        stored_name = source.get("stored_filename", "")
        stored_path = knowledge_dir / "originals" / Path(stored_name).name
        if not stored_path.is_file():
            source["ingestion_status"] = "error: stored original is missing"
            source["index_status"] = "not indexed"
        else:
            stored_content = stored_path.read_bytes()
            digest = sha256(stored_content).hexdigest()
            if digest != source.get("sha256"):
                source["ingestion_status"] = "error: stored original hash mismatch"
                source["index_status"] = "not indexed"
            elif source.get("source_class") == "illustrative_legacy":
                source["ingestion_status"] = "success"
                source["index_status"] = "excluded: illustrative content"
            elif not source.get("reviewed"):
                source["ingestion_status"] = "success"
                source["index_status"] = "pending local source review"
            else:
                try:
                    source["passage_count"] = len(
                        split_passages(
                            extract_document_text(
                                stored_content,
                                source["original_filename"],
                            )
                        )
                    )
                except (KeyError, KnowledgeBaseError) as error:
                    source["ingestion_status"] = f"error: {error}"
                    source["index_status"] = "not indexed"
                else:
                    source["ingestion_status"] = "success"
                    source["index_status"] = "indexed"
        sources.append(source)
    return sources


def load_indexable_passages(knowledge_dir: Path | str) -> List[Dict[str, Any]]:
    """Load reviewed, non-illustrative local documents into passage records."""
    knowledge_dir = Path(knowledge_dir)
    passages: List[Dict[str, Any]] = []
    for source in list_knowledge_sources(knowledge_dir):
        if source["ingestion_status"] != "success" or source["index_status"] != "indexed":
            continue
        file_path = knowledge_dir / "originals" / source["stored_filename"]
        content = file_path.read_bytes()
        actual_digest = sha256(content).hexdigest()
        if actual_digest != source["sha256"]:
            continue
        text = extract_document_text(content, source["original_filename"])
        source_passages = split_passages(text)
        for passage_number, passage in enumerate(source_passages, start=1):
            passages.append({
                "document_id": f"local:{source['source_id']}:{passage_number}",
                "source_id": source["source_id"],
                "title": source["source_title"],
                "category": source["source_class"],
                "content": passage,
                "source_title": source["source_title"],
                "source_url": source["source_url"],
                "source_ref": source["original_filename"],
                "original_filename": source["original_filename"],
                "publisher": source["publisher"],
                "published_date": source["published_date"],
                "updated_date": source["updated_date"],
                "retrieved_at": source["retrieved_at"],
                "ingested_at": source.get("ingested_at"),
                "document_type": source["document_type"],
                "source_class": source["source_class"],
                "review_status": "locally reviewed by uploader",
                "trusted": True,
            })
    return passages
