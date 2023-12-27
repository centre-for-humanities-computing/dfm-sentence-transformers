from typing import Dict, Iterable

import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange


def generate_cleaned_pairs(
    model: SentenceTransformer,
    dataset_name: str,
    dataset: Dataset,
    sentence1: str,
    sentence2: str,
    batch_size: int = 500,
    specificity: float = 1.2,
) -> Iterable[Dict]:
    """Cleans pairs based on consistency with a pretrained model.
    Assigns hard positives and hard negatives."""
    dataset = dataset["train"]
    n_batches = len(dataset) // batch_size
    for i_batch in trange(
        n_batches, desc=f"Processing all shards in {dataset_name}"
    ):
        batch = dataset.shard(num_shards=n_batches, index=i_batch)
        left_encodings = model.encode(batch[sentence1])
        right_encodings = model.encode(batch[sentence2])
        # N*N matrix of similarities of all left sentences and all right sentences.
        similarity_matrix = cosine_similarity(left_encodings, right_encodings)
        n = similarity_matrix.shape[0]
        # We select positive examples if their similarity is in the
        # 1/(N * specificity)-th quartile and lies on the diagonal
        # Aka. they have been assigned as a pair originally as well
        # 1/N is equal to the proportion of entries at the diagonal
        min_positive = np.quantile(
            np.ravel(similarity_matrix), 1 - (1 / (n * specificity))
        )
        diagonal = np.eye(n, dtype=bool)
        accepted_positive = (similarity_matrix > min_positive) & diagonal
        positive_indices = np.transpose(np.nonzero(accepted_positive))
        for i1, i2 in positive_indices:
            yield dict(
                sentence1=batch[sentence1][i1],
                sentence2=batch[sentence2][i2],
                label=1,
            )
        max_negative = np.quantile(
            np.ravel(similarity_matrix), 1 / (n * specificity)
        )
        # Only allow below the diagonal, so we don't get the same example twice
        lower = np.tri(similarity_matrix.shape[0], dtype=bool)
        accepted_negative = (
            (similarity_matrix < max_negative) & (~diagonal) & lower
        )
        negative_indices = np.transpose(np.nonzero(accepted_negative))
        # We want maximally as many negative examples as positive ones in a batch
        for i1, i2 in negative_indices[: positive_indices.shape[0]]:
            yield dict(
                sentence1=batch[sentence1][i1],
                sentence2=batch[sentence2][i2],
                label=0,
            )
