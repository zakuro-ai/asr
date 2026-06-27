"""Unit tests for the ETL online-download path (no network in CI).

The real 126 MB download + full ETL is exercised separately (integration), not
here — these tests cover URL resolution, idempotency, and the JSUT guard, which
are verifiable without hitting the network.
"""

import pytest

from asr_deepspeech.etl import JSUTDataset, LibriSpeechDataset


def test_url_for_known_subset():
    url = LibriSpeechDataset.url_for("dev-clean-2")
    assert url.startswith("https://www.openslr.org/")
    assert url.endswith("dev-clean-2.tar.gz")


def test_url_for_full_subset():
    assert LibriSpeechDataset.url_for("test-clean").endswith("12/test-clean.tar.gz")


def test_url_for_unknown_raises():
    with pytest.raises(ValueError):
        LibriSpeechDataset.url_for("not-a-subset")


def test_download_is_idempotent_when_already_present(tmp_path):
    """If the subset is already extracted, download() returns without hitting the network."""
    landing = tmp_path / "landing"
    (landing / "LibriSpeech" / "dev-clean-2").mkdir(parents=True)
    out = LibriSpeechDataset(16_000).download(str(landing), subset="dev-clean-2")
    assert out == str(landing)  # no archive downloaded, no exception


def test_jsut_download_explains_gdrive_requirement(tmp_path):
    """JSUT sits behind a private gdrive filestore — download() must say so, not fail silently."""
    with pytest.raises(NotImplementedError):
        JSUTDataset(16_000).download(str(tmp_path))
