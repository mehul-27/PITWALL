"""
Phase 5 — Merge, filter, and split dataset.
Combines template + Groq JSONL examples, deduplicates,
then splits by circuit into train.jsonl / val.jsonl / test.jsonl.
Also stores all Q&A pairs in ChromaDB for similarity retrieval during inference.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ALL_CIRCUITS,
    DATASET_DIR,
    SYSTEM_PROMPT,
    TEST_CIRCUITS,
    TEST_PATH,
    TRAIN_CIRCUITS,
    TRAIN_PATH,
    VAL_CIRCUITS,
    VAL_PATH,
)

log = logging.getLogger(__name__)

TEMPLATES_PATH = DATASET_DIR / "templates.jsonl"
EXPANDED_TEMPLATES_PATH = DATASET_DIR / "expanded_templates.jsonl"
CORRECTION_PATH = DATASET_DIR / "correction_examples.jsonl"
FAILURE_BOOST_PATH = DATASET_DIR / "failure_boost.jsonl"
GROQ_PATH      = DATASET_DIR / "groq.jsonl"
CHROMA_DIR     = DATASET_DIR / "chromadb"

_CIRCUIT_ALIASES: dict[str, str] = {
    "monza":            "Italy",
    "italian":          "Italy",
    "spa":              "Belgium",
    "belgian":          "Belgium",
    "silverstone":      "Great Britain",
    "british":          "Great Britain",
    "hungaroring":      "Hungary",
    "hungarian":        "Hungary",
    "marina bay":       "Singapore",
    "baku":             "Azerbaijan",
    "jeddah":           "Saudi Arabia",
    "interlagos":       "Sao Paulo",
    "são paulo":        "Sao Paulo",
    "sao paulo":        "Sao Paulo",
    "brazil":           "Sao Paulo",
    "brazilian":        "Sao Paulo",
    "imola":            "Emilia Romagna",
    "suzuka":           "Japan",
    "japanese":         "Japan",
    "zandvoort":        "Netherlands",
    "dutch":            "Netherlands",
    "albert park":      "Australia",
    "australian":       "Australia",
    "red bull ring":    "Austria",
    "austrian":         "Austria",
    "paul ricard":      "France",
    "french":           "France",
    "yas marina":       "Abu Dhabi",
    "losail":           "Qatar",
    "cota":             "United States",
    "istanbul":         "Turkey",
    "mugello":          "Tuscany",
    "nurburgring":      "Eifel",
    "portimao":         "Portugal",
    "algarve":          "Portugal",
    "sochi":            "Russia",
    "barcelona":        "Spain",
    "spanish":          "Spain",
    "las vegas":        "Las Vegas",
    "mexico city":      "Mexico City",
    "miami":            "Miami",
    "monaco":           "Monaco",
    "bahrain":          "Bahrain",
    "sakhir":           "Sakhir",
    "qatar":            "Qatar",
    "china":            "China",
    "shanghai":         "China",
    "canada":           "Canada",
    "montreal":         "Canada",
}


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, skip malformed lines."""
    examples: list[dict] = []
    if not path.exists():
        log.warning("File not found, skipping: %s", path)
        return examples
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d in %s: %s", i, path.name, exc)
    log.info("Loaded %d examples from %s", len(examples), path.name)
    return examples


def _get_user_content(example: dict) -> str:
    """Extract the first user message from an example."""
    for msg in example.get("messages", []):
        if msg.get("role") == "user":
            return msg["content"]
    return ""


def _get_assistant_content(example: dict) -> str:
    """Extract the first assistant message from an example."""
    for msg in example.get("messages", []):
        if msg.get("role") == "assistant":
            return msg["content"]
    return ""


def _deduplicate(examples: list[dict]) -> list[dict]:
    """Remove exact duplicates keyed on first user message content."""
    seen: set[str] = set()
    unique: list[dict] = []
    for ex in examples:
        key = _get_user_content(ex)
        if key in seen:
            continue
        seen.add(key)
        unique.append(ex)
    removed = len(examples) - len(unique)
    if removed:
        log.info("Removed %d duplicates, %d unique remain", removed, len(unique))
    return unique


def _tag_circuit(example: dict) -> str | None:
    """
    Extract the circuit name from user+assistant content.
    Matches against ALL_CIRCUITS (case-insensitive) and common aliases.
    Returns the canonical config.py circuit name, or None if unidentifiable.
    """
    text = " ".join(
        msg["content"] for msg in example.get("messages", [])
        if msg.get("role") in ("user", "assistant")
    ).lower()

    sorted_circuits = sorted(ALL_CIRCUITS, key=len, reverse=True)
    for circuit in sorted_circuits:
        if circuit.lower() in text:
            return circuit

    for alias, canonical in sorted(_CIRCUIT_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if alias in text:
            return canonical

    return None


def _split_by_circuit(
    examples: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split examples into train/val/test based on tagged circuit."""
    test_set  = set(TEST_CIRCUITS)
    val_set   = set(VAL_CIRCUITS)

    train: list[dict] = []
    val:   list[dict] = []
    test:  list[dict] = []
    untagged: list[dict] = []

    for ex in examples:
        circuit = ex.get("_circuit")
        if circuit is None:
            untagged.append(ex)
            continue
        if circuit in test_set:
            test.append(ex)
        elif circuit in val_set:
            val.append(ex)
        else:
            train.append(ex)

    if untagged:
        log.warning(
            "%d examples could not be tagged to a circuit — added to train",
            len(untagged),
        )
        train.extend(untagged)

    return train, val, test


def _verify_no_overlap(
    train: list[dict], val: list[dict], test: list[dict]
) -> None:
    """Confirm zero user-content overlap between splits."""
    train_keys = {_get_user_content(ex) for ex in train}
    val_keys   = {_get_user_content(ex) for ex in val}
    test_keys  = {_get_user_content(ex) for ex in test}

    tv = train_keys & val_keys
    tt = train_keys & test_keys
    vt = val_keys & test_keys

    if tv or tt or vt:
        log.error(
            "OVERLAP DETECTED  train∩val=%d  train∩test=%d  val∩test=%d",
            len(tv), len(tt), len(vt),
        )
        raise ValueError("Dataset splits have overlapping examples")

    log.info("Sanity check PASSED — zero overlap between splits")


def _strip_metadata(example: dict) -> dict:
    """Return a copy without internal metadata keys before writing to disk."""
    return {k: v for k, v in example.items() if not k.startswith("_")}


def _write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(_strip_metadata(ex), ensure_ascii=False) + "\n")
    log.info("Wrote %d examples -> %s", len(examples), path)


def _normalise_system_prompt(example: dict) -> dict:
    """Ensure every example has the canonical system prompt from config."""
    msgs = example.get("messages", [])
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = SYSTEM_PROMPT
    else:
        msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    example["messages"] = msgs
    return example


def _store_in_chromadb(examples: list[dict]) -> None:
    """
    Persist all Q&A pairs into a ChromaDB collection for similarity retrieval
    during inference. Each document is the user question; metadata stores the
    assistant answer and circuit tag.
    """
    try:
        import chromadb
    except ImportError:
        log.warning("chromadb not installed — skipping vector store step")
        return

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection_name = "pitwall_qa"
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 500
    ids:        list[str] = []
    documents:  list[str] = []
    metadatas:  list[dict] = []

    for i, ex in enumerate(examples):
        user_q     = _get_user_content(ex)
        assistant_a = _get_assistant_content(ex)
        circuit    = ex.get("_circuit", "unknown")

        if not user_q:
            continue

        ids.append(f"qa_{i}")
        documents.append(user_q)
        metadatas.append({
            "answer":  assistant_a,
            "circuit": circuit or "unknown",
        })

    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    log.info(
        "Stored %d Q&A pairs in ChromaDB -> %s  (collection: %s)",
        len(ids), CHROMA_DIR, collection_name,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("=" * 60)
    log.info("Phase 5 — Build dataset (merge, dedup, split, ChromaDB)")
    log.info("=" * 60)

    log.info("Test circuits:  %s", TEST_CIRCUITS)
    log.info("Val circuits:   %s", VAL_CIRCUITS)
    log.info("Train circuits: %d circuits", len(TRAIN_CIRCUITS))

    # 1. Load all sources
    templates          = _load_jsonl(TEMPLATES_PATH)
    expanded_templates = _load_jsonl(EXPANDED_TEMPLATES_PATH)
    corrections        = _load_jsonl(CORRECTION_PATH)
    failure_boost      = _load_jsonl(FAILURE_BOOST_PATH)
    groq               = _load_jsonl(GROQ_PATH)
    all_examples = templates + expanded_templates + corrections + failure_boost + groq
    log.info(
        "Combined: %d examples (templates=%d, expanded_templates=%d, corrections=%d, failure_boost=%d, groq=%d)",
        len(all_examples), len(templates), len(expanded_templates), len(corrections), len(failure_boost), len(groq),
    )

    # 2. Normalise system prompts
    all_examples = [_normalise_system_prompt(ex) for ex in all_examples]

    # 3. Deduplicate
    all_examples = _deduplicate(all_examples)

    # 4. Tag each example with its circuit
    untagged_count = 0
    for ex in all_examples:
        circuit = _tag_circuit(ex)
        ex["_circuit"] = circuit
        if circuit is None:
            untagged_count += 1
    log.info("Circuit tagging: %d tagged, %d untagged",
             len(all_examples) - untagged_count, untagged_count)

    # 5. Split by circuit
    train, val, test = _split_by_circuit(all_examples)

    # 6. Sanity check
    _verify_no_overlap(train, val, test)

    # 7. Write splits
    _write_jsonl(train, TRAIN_PATH)
    _write_jsonl(val,   VAL_PATH)
    _write_jsonl(test,  TEST_PATH)

    # 8. Final counts
    log.info("-" * 50)
    log.info("FINAL COUNTS")
    log.info("  Total:  %d", len(train) + len(val) + len(test))
    log.info("  Train:  %d", len(train))
    log.info("  Val:    %d", len(val))
    log.info("  Test:   %d", len(test))
    log.info("-" * 50)

    # 9. Store in ChromaDB for inference-time similarity retrieval
    _store_in_chromadb(all_examples)

    log.info("Done.")


if __name__ == "__main__":
    main()
