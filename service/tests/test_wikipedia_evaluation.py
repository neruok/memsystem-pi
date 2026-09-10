import bz2

import pytest

from memsystem.wikipedia_evaluation import _latency_metrics, load_wikipedia_chunks


def test_wikipedia_dump_streams_main_articles_and_skips_redirects(tmp_path):
    dump = tmp_path / "wiki.xml.bz2"
    dump.write_bytes(bz2.compress(b"""<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
      <page><title>Article</title><ns>0</ns><revision><text>useful text</text></revision></page>
      <page><title>Redirect</title><ns>0</ns><redirect title="Article"/><revision><text>#REDIRECT</text></revision></page>
      <page><title>Talk</title><ns>1</ns><revision><text>discussion</text></revision></page>
    </mediawiki>"""))

    chunks, articles = load_wikipedia_chunks(dump, 1)

    assert chunks == [("Article", "useful text")]
    assert articles == 1


def test_wikipedia_helpers_reject_missing_capacity_and_measure_latency(tmp_path):
    dump = tmp_path / "empty.xml.bz2"
    dump.write_bytes(bz2.compress(b"<mediawiki/>"))

    with pytest.raises(ValueError, match="only 0 eligible chunks"):
        load_wikipedia_chunks(dump, 1)
    assert _latency_metrics([0.001, 0.003]) == {
        "latency_ms_p50": 2.0,
        "latency_ms_p95": 2.9,
        "throughput_qps": 500.0,
    }
