"""Image-text retrieval evaluation for CLIP-style dual encoders.

Retrieval is the natural measurement for a contrastively trained dual encoder: it asks
whether the model puts the correct caption closest to each image (and vice versa) among all
candidates in the evaluation set. Recall@K and median rank are the standard metrics.

All metrics here are computed from real embeddings. There is no reference table of expected
values and nothing is estimated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RetrievalMetrics:
    """Measured retrieval metrics for one model on one evaluation set."""

    model_label: str
    samples: int
    #: image -> text
    i2t_recall_at_1: float = 0.0
    i2t_recall_at_5: float = 0.0
    i2t_recall_at_10: float = 0.0
    i2t_median_rank: float = 0.0
    i2t_mean_rank: float = 0.0
    #: text -> image
    t2i_recall_at_1: float = 0.0
    t2i_recall_at_5: float = 0.0
    t2i_recall_at_10: float = 0.0
    t2i_median_rank: float = 0.0
    t2i_mean_rank: float = 0.0
    #: Mean of the six recall figures, a single comparable summary number.
    mean_recall: float = 0.0
    encode_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _recall_and_ranks(similarity, dim: int):
    """Return (recall@1, recall@5, recall@10, median rank, mean rank) along ``dim``.

    ``similarity[i][j]`` is the score between item i of one modality and item j of the
    other; the ground-truth match is the diagonal.
    """
    import torch

    count = similarity.shape[0]
    targets = torch.arange(count, device=similarity.device)

    if dim == 0:
        scores = similarity
    else:
        scores = similarity.t()

    # Rank of the correct item = how many candidates score strictly higher, plus one.
    correct_scores = scores.gather(1, targets.view(-1, 1))
    ranks = (scores > correct_scores).sum(dim=1) + 1
    ranks_float = ranks.float()

    return (
        float((ranks <= 1).float().mean().item()),
        float((ranks <= 5).float().mean().item()),
        float((ranks <= 10).float().mean().item()),
        float(ranks_float.median().item()),
        float(ranks_float.mean().item()),
    )


def encode_dataset(
    model,
    processor,
    rows: list[dict],
    lmdb_dir,
    *,
    device: str = "cuda",
    batch_size: int = 32,
    image_size: int = 224,
    max_text_tokens: int = 77,
):
    """Encode images and captions into normalised embeddings.

    Returns ``(image_embeddings, text_embeddings, seconds)``.
    """
    import time

    import torch
    from PIL import Image

    from ml.datasets import LmdbImageReader

    reader = LmdbImageReader(lmdb_dir)
    model.eval()

    image_batches = []
    text_batches = []
    started = time.time()

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            images = []
            captions = []
            for row in chunk:
                rgb = reader.read_rgb(row["patch_id"])
                if rgb is None:
                    raise RuntimeError(f"unreadable image during evaluation: {row['patch_id']}")
                image = Image.fromarray(rgb, "RGB")
                if image.size != (image_size, image_size):
                    image = image.resize((image_size, image_size), Image.BICUBIC)
                images.append(image)
                captions.append(row["caption"])

            encoded = processor(
                text=captions,
                images=images,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_text_tokens,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}

            # PeftModel wraps CLIPModel; both expose the same feature methods.
            base = model.get_base_model() if hasattr(model, "get_base_model") else model
            image_features = base.get_image_features(pixel_values=encoded["pixel_values"])
            text_features = base.get_text_features(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
            )

            image_batches.append(
                (image_features / image_features.norm(dim=-1, keepdim=True)).float().cpu()
            )
            text_batches.append(
                (text_features / text_features.norm(dim=-1, keepdim=True)).float().cpu()
            )

    reader.close()
    seconds = time.time() - started
    return torch.cat(image_batches), torch.cat(text_batches), seconds


def evaluate_retrieval(
    model,
    processor,
    rows: list[dict],
    lmdb_dir,
    *,
    model_label: str,
    device: str = "cuda",
    batch_size: int = 32,
    image_size: int = 224,
    max_text_tokens: int = 77,
) -> RetrievalMetrics:
    """Compute bidirectional retrieval metrics for one model."""
    image_embeddings, text_embeddings, seconds = encode_dataset(
        model,
        processor,
        rows,
        lmdb_dir,
        device=device,
        batch_size=batch_size,
        image_size=image_size,
        max_text_tokens=max_text_tokens,
    )

    similarity = image_embeddings @ text_embeddings.t()

    i2t = _recall_and_ranks(similarity, dim=0)
    t2i = _recall_and_ranks(similarity, dim=1)

    metrics = RetrievalMetrics(
        model_label=model_label,
        samples=len(rows),
        i2t_recall_at_1=i2t[0],
        i2t_recall_at_5=i2t[1],
        i2t_recall_at_10=i2t[2],
        i2t_median_rank=i2t[3],
        i2t_mean_rank=i2t[4],
        t2i_recall_at_1=t2i[0],
        t2i_recall_at_5=t2i[1],
        t2i_recall_at_10=t2i[2],
        t2i_median_rank=t2i[3],
        t2i_mean_rank=t2i[4],
        encode_seconds=seconds,
    )
    metrics.mean_recall = (
        metrics.i2t_recall_at_1
        + metrics.i2t_recall_at_5
        + metrics.i2t_recall_at_10
        + metrics.t2i_recall_at_1
        + metrics.t2i_recall_at_5
        + metrics.t2i_recall_at_10
    ) / 6.0
    return metrics


def random_baseline_recall(samples: int) -> dict[str, float]:
    """Recall a uniformly random ranker would achieve, for context.

    Included so measured numbers can be read against chance rather than in isolation.
    """
    return {
        "recall_at_1": 1.0 / samples,
        "recall_at_5": min(1.0, 5.0 / samples),
        "recall_at_10": min(1.0, 10.0 / samples),
        "median_rank": samples / 2.0,
    }
