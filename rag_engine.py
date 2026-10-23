"""Evidence-only retrieval for passenger guidance backed by validated sources."""

import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from knowledge_base import KnowledgeBaseError, load_indexable_passages, list_knowledge_sources


class TransitRAGEngine:
    """Retrieve attributable passages without generating unsupported policy claims."""

    def __init__(
        self,
        data_path: Optional[str] = None,
        gtfs_feed: Optional[Any] = None,
        evidence_documents: Optional[Sequence[Dict[str, Any]]] = None,
        index_error: Optional[str] = None,
        knowledge_dir: Optional[str | Path] = None,
    ) -> None:
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.data_path = data_path or os.path.join(
            project_root, "data", "travel_guides.csv"
        )
        self.knowledge_dir = Path(knowledge_dir or Path(project_root) / "data" / "passenger_knowledge")
        self.knowledge_source_error: Optional[str] = None
        self.legacy_document_count = 0
        self.legacy_source_status = ""
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.documents: List[Dict[str, Any]] = []
        self.is_indexed = False
        self.index_error = index_error
        self.model_available = False
        self.model_status = (
            "No language-model adapter is configured. Answers use retrieved evidence "
            "only; no model-generated answer is claimed."
        )

        self._inspect_legacy_guides()
        if evidence_documents is not None:
            trusted_documents = list(evidence_documents)
        elif gtfs_feed is not None:
            trusted_documents = self._build_gtfs_evidence(gtfs_feed)
        else:
            trusted_documents = []
            if self.index_error is None:
                self.index_error = (
                    "No validated transport feed is available to provide attributable "
                    "passenger-advisor evidence."
                )
        try:
            trusted_documents.extend(load_indexable_passages(self.knowledge_dir))
        except KnowledgeBaseError as error:
            self.knowledge_source_error = str(error)
        self._initialize_index(trusted_documents)

    def _inspect_legacy_guides(self) -> None:
        """Count, but never index, the unauthenticated illustrative guide records."""
        if not os.path.isfile(self.data_path):
            self.legacy_source_status = "travel_guides.csv is not present."
            return
        try:
            frame = pd.read_csv(self.data_path, usecols=["guide_id", "title", "content"])
        except ValueError as error:
            self.legacy_source_status = (
                "travel_guides.csv could not be audited because required columns are "
                f"missing: {error}"
            )
            return
        except (OSError, pd.errors.ParserError, UnicodeError) as error:
            self.legacy_source_status = (
                f"travel_guides.csv could not be audited: {error}"
            )
            return
        self.legacy_document_count = len(frame)
        self.legacy_source_status = (
            f"{self.legacy_document_count} local guide record(s) excluded from answer "
            "evidence because they have no source URL or publication/update metadata "
            "and include illustrative/simulated claims."
        )

    @staticmethod
    def _build_gtfs_evidence(feed: Any) -> List[Dict[str, Any]]:
        """Create short evidence passages from validated feed metadata and behavior."""
        metadata = feed.source_metadata
        snapshot_id = metadata.get("archive_snapshot_id", "snapshot ID unavailable")
        source_url = metadata.get("source_url", "")
        retrieved_at = metadata.get("retrieved_at_utc", "")
        source_title = "TGSRTC archived static GTFS timetable"
        source_reference = (
            f"MobilityDatabase snapshot {snapshot_id}; "
            f"GTFS feed version {feed.feed_info['feed_version']}"
        )
        provenance = {
            "source_title": source_title,
            "source_url": source_url,
            "source_ref": source_reference,
            "source_id": f"tgsrtc:{snapshot_id}",
            "publisher": feed.feed_info.get("feed_publisher_name", "TGSRTC"),
            "source_class": "official_tgsrtc",
            "document_type": "Static GTFS timetable",
            "original_filename": "tgsrtc_gtfs.zip",
            "review_status": "validated local archive metadata",
            "published_date": None,
            "updated_date": None,
            "retrieved_at": retrieved_at,
            "source_kind": "validated static GTFS feed",
            "trusted": True,
        }
        coverage_content = (
            f"The validated TGSRTC static GTFS feed covers service dates "
            f"{feed.feed_start_date.isoformat()} through "
            f"{feed.feed_end_date.isoformat()}. The agency timezone is "
            f"{feed.agency['agency_timezone']}. Its feed version is "
            f"{feed.feed_info['feed_version']}. Archive retrieval time recorded by "
            f"this application: {retrieved_at or 'not recorded'}. These are archived "
            "timetable records, not a real-time operating-status feed."
        )
        coverage_document = {
            **provenance,
            "document_id": "tgsrtc-feed-coverage",
            "category": "Verified timetable coverage",
            "title": "TGSRTC timetable coverage and timezone",
            "content": coverage_content,
        }

        adapter_content = (
            "RouteRescue bus candidates are built from an active service-calendar "
            "date and a single trip's ordered stop_times records. The selected origin "
            "must occur before the destination in that trip. Scheduled departure must "
            "be at or after the requested local time. Journey duration is calculated "
            "from the selected origin departure time to the destination arrival time; "
            "GTFS times after midnight are interpreted relative to the service date. "
            "The adapter creates direct, same-trip candidates only and does not "
            "construct transfers. Reference: tgsrtc_gtfs.py, "
            "TgsrtcGtfsFeed.search_journeys and _make_journey."
        )
        adapter_document = {
            **provenance,
            "source_id": "route-rescue-gtfs-adapter",
            "source_title": "RouteRescue AI TGSRTC GTFS adapter",
            "source_url": "",
            "source_ref": "tgsrtc_gtfs.py: TgsrtcGtfsFeed.search_journeys and _make_journey",
            "publisher": "RouteRescue AI",
            "source_class": "application_reference",
            "document_type": "Application implementation reference",
            "original_filename": "tgsrtc_gtfs.py",
            "review_status": "local application reference",
            "published_date": None,
            "updated_date": None,
            "retrieved_at": None,
            "source_kind": "application implementation reference",
            "document_id": "route-rescue-gtfs-search",
            "category": "How to interpret scheduled results",
            "title": "How RouteRescue builds a scheduled bus journey",
            "content": adapter_content,
        }

        absent_features = []
        for filename, description in (
            ("fare_attributes.txt", "fare tables"),
            ("fare_rules.txt", "fare rules"),
            ("shapes.txt", "route shapes"),
        ):
            if not (feed.feed_dir / filename).is_file():
                absent_features.append(description)
        limitation_items = []
        if absent_features:
            limitation_items.append(
                f"The inspected feed does not include {', '.join(absent_features)}."
            )
        if metadata.get("static_data_only") is True:
            limitation_items.append(
                "The source metadata identifies this archive as static data; it does "
                "not establish current bus operation, current vehicle location, live "
                "delays, or cancellations. It does not provide real-time bus positions "
                "or confirm whether a scheduled service is operating now."
            )
        limitation_items.append(
            "The application does not construct transfers between separate trips."
        )
        limitations_document = {
            **provenance,
            "document_id": "tgsrtc-feed-limitations",
            "category": "Verified source limitations",
            "title": "What this TGSRTC timetable cannot establish",
            "content": " ".join(limitation_items),
        }
        return [coverage_document, adapter_document, limitations_document]

    def _initialize_index(self, documents: Sequence[Dict[str, Any]]) -> None:
        required_fields = {
            "document_id",
            "title",
            "category",
            "content",
            "source_title",
            "source_ref",
            "trusted",
        }
        seen_document_ids = set()
        for document in documents:
            if not required_fields.issubset(document):
                self.index_error = (
                    "A trusted evidence document is missing required citation fields."
                )
                return
            if document["trusted"] is not True:
                continue
            document_id = document["document_id"]
            if not isinstance(document_id, str) or not document_id.strip():
                self.index_error = "A trusted evidence document has no valid document ID."
                return
            if document_id in seen_document_ids:
                self.index_error = (
                    "Trusted evidence documents have duplicate document IDs."
                )
                return
            seen_document_ids.add(document_id)
            if not isinstance(document["source_title"], str) or not document[
                "source_title"
            ].strip():
                self.index_error = (
                    "A trusted evidence document has no valid citation source title."
                )
                return
            if not isinstance(document["content"], str) or not document["content"].strip():
                self.index_error = "A trusted evidence document has empty content."
                return
            document.setdefault("publisher", document["source_title"])
            document.setdefault("source_class", "general_reference")
            document.setdefault("document_type", document["category"])
            document.setdefault("original_filename", document.get("source_ref", ""))
            document.setdefault("review_status", "source status not supplied")
            source_ref = document["source_ref"]
            source_url = document.get("source_url", "")
            if not isinstance(source_ref, str) or not isinstance(source_url, str):
                self.index_error = (
                    "A trusted evidence document has invalid citation metadata."
                )
                return
            if not source_ref.strip() and not source_url.strip():
                self.index_error = (
                    "A trusted evidence document has no URL or document reference."
                )
                return

        self.documents = [doc for doc in documents if doc.get("trusted") is True]
        if not self.documents:
            if self.index_error is None:
                self.index_error = (
                    "No attributable, validated evidence documents are available."
                )
            return

        corpus = [
            f"{document['title']} {document['category']} "
            f"{document['content']}"
            for document in self.documents
        ]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=2500,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_indexed = True
        self.index_error = None

    def source_registry(self) -> List[Dict[str, Any]]:
        """Return indexed built-in sources and local ingestion statuses."""
        grouped_records: Dict[str, Dict[str, Any]] = {}
        for document in self.documents:
            if str(document.get("document_id", "")).startswith("local:"):
                continue
            group_key = str(document.get("source_id") or document["document_id"])
            if group_key not in grouped_records:
                grouped_records[group_key] = {
                    "Source title": document["source_title"],
                    "Publisher": document.get("publisher", "not provided"),
                    "Source class": document.get("source_class", "not provided"),
                    "Document type": document.get("document_type", "not provided"),
                    "Source URL / file": (
                        document.get("source_url")
                        or document.get("original_filename")
                        or document.get("source_ref")
                        or "not provided"
                    ),
                    "Retrieved": document.get("retrieved_at") or "not recorded",
                    "Registered": document.get("ingested_at") or "not applicable",
                    "Publication date": document.get("published_date") or "not provided",
                    "Update date": document.get("updated_date") or "not provided",
                    "Status": "Indexed",
                    "Passages": 0,
                    "Review": document.get("review_status", "not provided"),
                }
            grouped_records[group_key]["Passages"] += 1
        records: List[Dict[str, Any]] = list(grouped_records.values())
        try:
            for source in list_knowledge_sources(self.knowledge_dir):
                records.append({
                    "Source title": source.get("source_title", "Untitled local source"),
                    "Publisher": source.get("publisher", "not provided"),
                    "Source class": source.get("source_class", "not provided"),
                    "Document type": source.get("document_type", "not provided"),
                    "Source URL / file": (
                        source.get("source_url")
                        or source.get("original_filename")
                        or "not provided"
                    ),
                    "Retrieved": source.get("retrieved_at", "not recorded"),
                    "Registered": source.get("ingested_at", "not recorded"),
                    "Publication date": source.get("published_date") or "not provided",
                    "Update date": source.get("updated_date") or "not provided",
                    "Status": (
                        f"{source['ingestion_status']}; {source['index_status']}"
                    ),
                    "Passages": (
                        source.get("passage_count", 0)
                        if source["index_status"] == "indexed"
                        else 0
                    ),
                    "Review": (
                        "locally reviewed by uploader; not independently authenticated"
                        if source.get("reviewed")
                        else "not reviewed"
                    ),
                })
        except KnowledgeBaseError as error:
            self.knowledge_source_error = str(error)
        return records

    def search(self, query: str, top_k: int = 3, min_score: float = 0.12) -> List[Dict[str, Any]]:
        """Retrieve only relevant passages from trusted evidence."""
        return self.search_with_evidence(query, top_k, min_score)["results"]

    def search_with_evidence(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.12,
    ) -> Dict[str, Any]:
        if not self.is_indexed:
            return self._no_evidence(
                self.index_error or "The evidence index is unavailable.",
                status="Error",
            )
        if not isinstance(query, str) or not query.strip():
            return self._no_evidence("Enter a passenger question to search.")
        if top_k < 1:
            return self._no_evidence("The retrieval result limit must be positive.", status="Error")

        try:
            query_vector = self.vectorizer.transform([query.strip()])
            scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        except (ValueError, TypeError, AttributeError) as error:
            return self._no_evidence(
                f"Evidence retrieval failed: {error}",
                status="Error",
            )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), self.documents[index]["document_id"]),
        )
        results = []
        for index in ranked_indices:
            score = float(scores[index])
            if score < min_score or score <= 0:
                break
            document = self.documents[index]
            results.append({
                **document,
                "score": round(score, 4),
                "relevance_pct": f"{min(100, int(score * 100))}%",
            })
            if len(results) >= top_k:
                break

        if not results:
            return self._no_evidence(
                "The available sources do not establish an answer to this question."
            )

        conflict_pair = self._find_potential_conflict(results, min_score)
        if conflict_pair:
            conflict_results = [conflict_pair[0], conflict_pair[1]]
            return {
                "has_evidence": False,
                "confidence_level": "Conflicting evidence",
                "explanation": (
                    "Two locally indexed passages appear to make opposing claims "
                    "about overlapping terms. The advisor cannot resolve the conflict."
                ),
                "answer": (
                    "The indexed sources may conflict on this point. The available "
                    "evidence does not support a resolved answer."
                ),
                "answer_mode": "Abstained because potentially conflicting passages were retrieved.",
                "results": conflict_results,
                "supporting_sources": [
                    result["source_url"] or result["source_ref"]
                    for result in conflict_results
                ],
            }

        return {
            "has_evidence": True,
            "confidence_level": "Retrieved evidence",
            "explanation": f"Retrieved {len(results)} relevant trusted evidence passage(s).",
            "answer": results[0]["content"],
            "answer_mode": "Evidence-only extractive answer; no language model was used.",
            "results": results,
            "supporting_sources": [
                result["source_url"] or result["source_ref"] for result in results
            ],
        }

    @staticmethod
    def _find_potential_conflict(
        results: Sequence[Dict[str, Any]],
        min_score: float,
    ) -> Optional[tuple[Dict[str, Any], Dict[str, Any]]]:
        negation_terms = {
            "no",
            "not",
            "never",
            "without",
            "cannot",
            "cant",
            "doesnt",
            "isnt",
            "arent",
            "wont",
            "shouldnt",
        }

        def sentences(text: str) -> List[str]:
            return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]

        def terms(text: str) -> set[str]:
            return {
                term
                for term in re.findall(r"[a-z0-9]+", text.lower())
                if term not in ENGLISH_STOP_WORDS and term not in negation_terms
            }

        for left_index, left in enumerate(results):
            left_source = (
                left.get("source_id")
                or left.get("source_url")
                or left.get("source_ref")
            )
            for right in results[left_index + 1:]:
                right_source = (
                    right.get("source_id")
                    or right.get("source_url")
                    or right.get("source_ref")
                )
                if not left_source or left_source == right_source:
                    continue
                conflict_score_floor = max(min_score, 0.18)
                if (
                    float(left.get("score", 0)) < conflict_score_floor
                    or float(right.get("score", 0)) < conflict_score_floor
                ):
                    continue
                for left_sentence in sentences(left["content"]):
                    left_terms = terms(left_sentence)
                    if len(left_terms) < 2:
                        continue
                    left_negated = bool(
                        re.search(
                            r"\b(?:no|not|never|without|cannot|can't|doesn't|isn't|aren't|won't|shouldn't)\b",
                            left_sentence,
                            re.IGNORECASE,
                        )
                    )
                    for right_sentence in sentences(right["content"]):
                        right_terms = terms(right_sentence)
                        if len(right_terms) < 2 or left_negated == bool(
                            re.search(
                                r"\b(?:no|not|never|without|cannot|can't|doesn't|isn't|aren't|won't|shouldn't)\b",
                                right_sentence,
                                re.IGNORECASE,
                            )
                        ):
                            continue
                        overlap = left_terms & right_terms
                        if len(overlap) >= 3:
                            return left, right
        return None

    @staticmethod
    def _no_evidence(explanation: str, status: str = "No Evidence Found") -> Dict[str, Any]:
        return {
            "has_evidence": False,
            "confidence_level": status,
            "explanation": explanation,
            "answer": "The available sources do not establish an answer to this question.",
            "answer_mode": "Evidence-only mode; no language model was used.",
            "results": [],
            "supporting_sources": [],
        }
