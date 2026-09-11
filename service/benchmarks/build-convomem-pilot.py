#!/usr/bin/env python3
"""Build a bounded message-retrieval pilot from Salesforce ConvoMem."""

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import random
from urllib.request import urlopen

import pyarrow as arrow
import pyarrow.parquet as parquet

REVISION = "e3e9b39115b02346824c70d349350de738f8be41"
PERSONAS = (
    "0050e213-5032-42a0-8041-b5eef2f8ab91_Telemarketer.json",
    "03d6ce14-747d-43cb-8b31-d9c73391f58c_Technical_Support_Engineer_Tier_1.json",
    "0b4fb858-701e-4b33-972f-170cc36ffd12_Partner_Success_Manager.json",
)
TASKS = {
    "user_evidence": "user_evidence/1_evidence",
    "assistant_facts": "assistant_facts_evidence/1_evidence",
    "changing_evidence": "changing_evidence/2_evidence",
    "preference_evidence": "preference_evidence/1_evidence",
    "implicit_connection": "implicit_connection_evidence/1_evidence",
}
BASE = f"https://huggingface.co/datasets/Salesforce/ConvoMem/resolve/{REVISION}/core_benchmark"


def _download(path: str) -> dict:
    with urlopen(f"{BASE}/{path}") as response:
        return json.load(response)


def _messages(documents, prefix, persona, conversations):
    indexed = []
    for conversation_index, conversation in enumerate(conversations):
        for message_index, message in enumerate(conversation["messages"]):
            identifier = f"{prefix}-{conversation_index}-{message_index}"
            row = (identifier, message["speaker"], message["text"], persona)
            documents.append(row)
            indexed.append(row)
    return indexed


def _relevant_messages(indexed, evidence):
    relevant = []
    for message in evidence:
        evidence_text = " ".join(message["text"].casefold().split())
        candidates = [
            (identifier, " ".join(text.casefold().split()))
            for identifier, speaker, text, _ in indexed
            if speaker.casefold() == message["speaker"].casefold()
        ]
        exact = [identifier for identifier, text in candidates if evidence_text in text]
        if len(exact) == 1:
            relevant.extend(exact)
            continue
        if len(exact) > 1:
            raise ValueError("message evidence matched multiple conversation messages")
        windows = sorted((
            (
                SequenceMatcher(
                    None, evidence_text,
                    " ".join(text for _, text in candidates[start:end]),
                ).ratio(),
                [identifier for identifier, _ in candidates[start:end]],
            )
            for start in range(len(candidates))
            for end in range(start + 2, min(start + 3, len(candidates)) + 1)
            if len({identifier.rsplit("-", 1)[0] for identifier, _ in candidates[start:end]}) == 1
        ), reverse=True)
        if (
            windows
            and windows[0][0] >= 0.9
            and (len(windows) == 1 or windows[0][0] - windows[1][0] >= 0.02)
        ):
            relevant.extend(windows[0][1])
            continue
        ranked = sorted(
            (
                (SequenceMatcher(None, evidence_text, text).ratio(), identifier)
                for identifier, text in candidates
            ),
            reverse=True,
        )
        if (
            not ranked or ranked[0][0] < 0.9
            or (len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.1)
        ):
            raise ValueError("message evidence did not match one unique conversation message")
        relevant.append(ranked[0][1])
    return list(dict.fromkeys(relevant))


def build(
    output: Path,
    queries_per_task: int,
    filler_conversations: int,
    seed: int,
    personas: tuple[str, ...] = PERSONAS,
) -> None:
    if queries_per_task < 1 or filler_conversations < 1:
        raise ValueError("query and filler counts must be positive")
    if not personas or len(personas) != len(set(personas)):
        raise ValueError("personas must be nonempty and unique")
    output.mkdir(parents=True, exist_ok=True)
    documents = []
    query_rows = {task: [] for task in TASKS}
    qrel_rows = {task: [] for task in TASKS}
    query_rows["changing_evidence_history"] = []
    qrel_rows["changing_evidence_history"] = []
    randomizer = random.Random(seed)
    skipped = 0

    for persona_index, persona in enumerate(personas):
        persona_id = persona.removesuffix(".json")
        filler = _download(f"filler_conversations/{persona}")["evidence_items"]
        if filler_conversations > len(filler):
            raise ValueError(f"requested more filler conversations than available: {persona}")
        for index, item in enumerate(randomizer.sample(filler, filler_conversations)):
            _messages(documents, f"p{persona_index}-filler-{index}", persona_id, item["conversations"])

        for task, source in TASKS.items():
            items = _download(f"evidence_questions/{source}/{persona}")["evidence_items"]
            if queries_per_task > len(items):
                raise ValueError(f"requested more {task} queries than available: {persona}")
            randomizer.shuffle(items)
            accepted = 0
            for item in items:
                query_id = f"p{persona_index}-{task}-{accepted}"
                item_documents = []
                indexed = _messages(
                    item_documents, query_id, persona_id, item["conversations"]
                )
                try:
                    relevant = _relevant_messages(indexed, item["message_evidences"])
                except ValueError:
                    skipped += 1
                    continue
                documents.extend(item_documents)
                query_rows[task].append((query_id, item["question"], item["answer"], persona_id))
                selected = relevant[-1:] if task == "changing_evidence" else relevant
                qrel_rows[task].extend((query_id, identifier, 1) for identifier in selected)
                if task == "changing_evidence":
                    history_id = f"{query_id}-history"
                    query_rows["changing_evidence_history"].append(
                        (history_id, item["question"], item["answer"], persona_id)
                    )
                    qrel_rows["changing_evidence_history"].extend(
                        (history_id, identifier, 1) for identifier in relevant
                    )
                accepted += 1
                if accepted == queries_per_task:
                    break
            if accepted < queries_per_task:
                raise ValueError(f"not enough matchable {task} queries: {persona}")

    for task, queries in query_rows.items():
        parquet.write_table(arrow.table({
            "id": [row[0] for row in queries],
            "text": [row[1] for row in queries],
            "answer": [row[2] for row in queries],
            "persona": [row[3] for row in queries],
        }), output / f"{task}-queries.parquet")
        qrels = qrel_rows[task]
        parquet.write_table(arrow.table({
            "query-id": [row[0] for row in qrels],
            "corpus-id": [row[1] for row in qrels],
            "score": [row[2] for row in qrels],
        }), output / f"{task}-qrels.parquet")

    parquet.write_table(arrow.table({
        "id": [row[0] for row in documents],
        "title": [row[1] for row in documents],
        "text": [row[2] for row in documents],
        "persona": [row[3] for row in documents],
    }), output / "corpus.parquet")
    (output / "NAME").write_text("Salesforce/ConvoMem pooled-persona message retrieval\n")
    (output / "REVISION").write_text(
        f"{REVISION}-personas-{len(personas)}-q{queries_per_task}-f{filler_conversations}-s{seed}\n"
    )
    print(json.dumps({
        "documents": len(documents),
        "queries": sum(map(len, query_rows.values())),
        "personas": len(personas),
        "skipped_unmatched_queries": skipped,
        "tasks": list(query_rows),
        "output": str(output),
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/convomem-pilot"))
    parser.add_argument("--queries-per-task", type=int, default=20)
    parser.add_argument("--filler-conversations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--personas", nargs="+", default=list(PERSONAS))
    args = parser.parse_args()
    build(
        args.output,
        args.queries_per_task,
        args.filler_conversations,
        args.seed,
        tuple(args.personas),
    )


if __name__ == "__main__":
    main()
