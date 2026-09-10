import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from memsystem.longmemeval_scale_evaluation import load_corpus


def test_longmemeval_corpus_loads_bounded_clean_text(tmp_path):
    path = tmp_path / "corpus.parquet"
    parquet.write_table(arrow.table({
        "title": ["First", None, "Third"],
        "text": ["one", "two\x00clean", "three"],
    }), path)

    assert load_corpus(path, 2) == [
        ("First", "one"),
        ("LongMemEval", "twoclean"),
    ]
    with pytest.raises(ValueError, match="only 3 documents"):
        load_corpus(path, 4)
