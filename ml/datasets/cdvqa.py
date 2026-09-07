"""CDVQA dataset adapter — loads the real CDVQA JSON splits and joins them to SECOND imagery.

CDVQA (Yuan et al., 2022) ships question/answer/image JSON only; the referenced image
`file_name`s are SECOND dataset images (0.1524 m). This module joins questions → images →
answers into flat samples, builds a closed answer vocabulary and a question word vocabulary
**from the train split only** (no leakage), and exposes split-integrity checks.

It performs NO image loading itself (features are cached separately) and NO training — it is the
single source of truth for how the real dataset is parsed.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass
from pathlib import Path

PAD, UNK = "<pad>", "<unk>"
SPLITS = ("Train", "Val", "Test", "Test2")


@dataclass(frozen=True)
class CDVQASample:
    question_id: int
    img_file: str  # e.g. "10589.png" (present in SECOND im1/ and im2/)
    question: str
    answer: str  # lowercased closed-vocab token
    qtype: str


def load_split(base_dir: str | Path, split: str) -> list[CDVQASample]:
    """Load and join one CDVQA split into flat samples.

    Raises FileNotFoundError if the split JSONs are absent (caller must report unavailable
    rather than substitute anything).
    """
    base = Path(base_dir)
    q = json.loads((base / f"{split}_questions.json").read_text())["questions"]
    a = json.loads((base / f"{split}_answers.json").read_text())["answers"]
    im = json.loads((base / f"{split}_images.json").read_text())["images"]

    img_by_id = {rec["id"]: rec["file_name"] for rec in im}
    answer_by_qid = {rec["question_id"]: str(rec["answer"]).strip().lower() for rec in a}

    samples: list[CDVQASample] = []
    for rec in q:
        qid = rec["id"]
        ans = answer_by_qid.get(qid)
        img = img_by_id.get(rec["img_id"])
        if ans is None or img is None:
            continue  # unjoinable record; skipped and counted by the caller
        samples.append(
            CDVQASample(
                question_id=qid,
                img_file=img,
                question=str(rec["question"]).strip(),
                answer=ans,
                qtype=str(rec.get("type", "unknown")),
            )
        )
    return samples


def build_answer_vocab(train: list[CDVQASample]) -> dict[str, int]:
    """Closed answer vocabulary, ordered by frequency (train only)."""
    counts = collections.Counter(s.answer for s in train)
    return {ans: i for i, (ans, _) in enumerate(counts.most_common())}


def build_word_vocab(train: list[CDVQASample], min_freq: int = 1) -> dict[str, int]:
    """Question word vocabulary from the train split only."""
    counts = collections.Counter()
    for s in train:
        counts.update(_tokens(s.question))
    vocab = {PAD: 0, UNK: 1}
    for word, freq in counts.most_common():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def _tokens(question: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() else " " for c in question).split()


def encode_question(question: str, word_vocab: dict[str, int], max_len: int = 20) -> list[int]:
    ids = [word_vocab.get(t, word_vocab[UNK]) for t in _tokens(question)][:max_len]
    ids += [word_vocab[PAD]] * (max_len - len(ids))
    return ids


def unique_images(samples: list[CDVQASample]) -> set[str]:
    return {s.img_file for s in samples}


def check_leakage(splits: dict[str, list[CDVQASample]]) -> dict[str, int]:
    """Return pairwise image overlaps; Train/Val/Test must be disjoint (0)."""
    imgs = {name: unique_images(s) for name, s in splits.items()}
    report: dict[str, int] = {}
    pairs = [("Train", "Val"), ("Train", "Test"), ("Val", "Test"), ("Train", "Test2")]
    for a, b in pairs:
        if a in imgs and b in imgs:
            report[f"{a}_vs_{b}"] = len(imgs[a] & imgs[b])
    return report
