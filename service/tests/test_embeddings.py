import numpy as np
import pytest

from memsystem.embeddings import ACTIVE_PROFILE, normalize_embedding


@pytest.mark.parametrize(
    "values",
    (
        [],
        [0.0] * ACTIVE_PROFILE.embedding_dimension,
        [float("nan")] * ACTIVE_PROFILE.embedding_dimension,
    ),
)
def test_embedding_rejects_invalid_vectors(values):
    with pytest.raises(ValueError, match="embedding"):
        normalize_embedding(values)


def test_embedding_is_contiguous_float32_unit_vector():
    vector = normalize_embedding([1.0] * ACTIVE_PROFILE.embedding_dimension)

    assert vector.dtype == np.float32
    assert vector.flags.c_contiguous
    assert np.linalg.norm(vector) == pytest.approx(1.0)
