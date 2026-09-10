import hashlib
from uuid import uuid4

import pytest
from turbovec import IdMapIndex

from memsystem.embeddings import ACTIVE_PROFILE
from memsystem.vector_index import TurboVecIndex


def test_turbovec_index_round_trip(tmp_path):
    path = tmp_path / "generation-1.tvim"
    owner = TurboVecIndex(uuid4(), path)
    vector = [1.0] + [0.0] * (ACTIVE_PROFILE.embedding_dimension - 1)

    owner.add(42, vector)
    owner.add_many([43, 44], [vector, vector])
    checksum = owner.sync()
    loaded = IdMapIndex.load(str(path))

    assert checksum == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(loaded.contains(identifier) for identifier in (42, 43, 44))
    scores, ids = owner.search(vector, 1, [42])
    assert ids.tolist() == [[42]]
    assert scores[0, 0] > 0.9
    with pytest.raises(ValueError):
        owner.search(vector, 1, [])

    owner.remove(42)
    owner.sync()

    assert not IdMapIndex.load(str(path)).contains(42)
    owner.close()


def test_turbovec_index_poisoned_after_reload_failure(tmp_path):
    path = tmp_path / "generation-1.tvim"
    owner = TurboVecIndex(uuid4(), path)
    owner.add(42, [1.0] + [0.0] * (ACTIVE_PROFILE.embedding_dimension - 1))
    owner.sync()
    path.write_bytes(b"broken")

    with pytest.raises(OSError):
        owner.reload()
    with pytest.raises(RuntimeError, match="reconstructed"):
        owner.remove(42)

    assert not owner.healthy
    owner.close()


def test_turbovec_index_allows_only_one_owner(tmp_path):
    path = tmp_path / "generation-1.tvim"
    owner = TurboVecIndex(uuid4(), path)

    with pytest.raises(BlockingIOError):
        TurboVecIndex(uuid4(), path)

    owner.close()
    successor = TurboVecIndex(uuid4(), path)
    vector = [1.0] + [0.0] * (ACTIVE_PROFILE.embedding_dimension - 1)
    with pytest.raises(RuntimeError, match="reconstructed"):
        owner.add(42, vector)
    with pytest.raises(RuntimeError, match="reconstructed"):
        owner.remove(42)
    with pytest.raises(RuntimeError, match="reconstructed"):
        owner.search(vector, 1, [42])
    with pytest.raises(RuntimeError, match="reconstructed"):
        owner.sync()
    with pytest.raises(RuntimeError, match="reconstructed"):
        owner.checksum()
    with pytest.raises(RuntimeError, match="closed"):
        owner.reload()
    successor.close()
