import tarfile

import numpy as np
import pytest
import soundfile as sf

from asr_deepspeech.data.dataset import (
    ManifestDataset,
    build_vocabulary,
    manifest_loader,
    read_clip,
)


def _write_wav(path, seconds, freq):
    t = np.linspace(0, seconds, int(16000 * seconds), endpoint=False, dtype=np.float32)
    sf.write(str(path), 0.3 * np.sin(2 * np.pi * freq * t), 16000, subtype="PCM_16")


@pytest.fixture
def shard_root(tmp_path):
    """One tar shard with 3 clips + the manifest rows that address them by byte span."""
    clips = {"a.wav": (0.4, 300), "b.wav": (0.6, 500), "c.wav": (0.5, 700)}
    for name, (sec, hz) in clips.items():
        _write_wav(tmp_path / name, sec, hz)
    shard = tmp_path / "audio" / "train-000.tar"
    shard.parent.mkdir()
    with tarfile.open(shard, "w") as tf:
        for name in clips:
            tf.add(tmp_path / name, arcname=name)
    rows = []
    with tarfile.open(shard) as tf:
        for info, text in zip(tf.getmembers(), ["ab", "ba ab", "cc"]):
            rows.append(
                {
                    "audio": f"audio/train-000.tar#{info.name}",
                    "transcript": text,
                    "offset": info.offset_data,
                    "size": info.size,
                }
            )
    return tmp_path, rows


def test_read_clip_by_byte_span_equals_the_file(shard_root):
    root, rows = shard_root
    for row, name in zip(rows, ["a.wav", "b.wav", "c.wav"]):
        got = read_clip(root, row["audio"], row["offset"], row["size"])
        want, _ = sf.read(str(root / name), dtype="float32")
        assert np.array_equal(got, want)


def test_read_clip_without_span_scans_the_tar(shard_root):
    root, rows = shard_root
    got = read_clip(root, rows[1]["audio"])
    want, _ = sf.read(str(root / "b.wav"), dtype="float32")
    assert np.array_equal(got, want)


def test_read_clip_plain_path_and_sample_rate_check(tmp_path):
    _write_wav(tmp_path / "x.wav", 0.2, 400)
    assert read_clip(tmp_path, "x.wav").shape == (3200,)
    sf.write(str(tmp_path / "y.wav"), np.zeros(800, dtype="float32"), 8000)
    with pytest.raises(ValueError, match="16000"):
        read_clip(tmp_path, "y.wav")


def test_vocabulary_has_blank_first_and_is_deterministic():
    v = build_vocabulary(["ba ab", "cc"])
    assert v == ["_", " ", "a", "b", "c"]
    assert build_vocabulary(["cc", "ba ab"]) == v
    with pytest.raises(ValueError):
        build_vocabulary(["a_b"])


def test_dataset_item_and_loader_batch(shard_root, audio_conf):
    root, rows = shard_root
    vocab = build_vocabulary([r["transcript"] for r in rows])
    labels = {c: i for i, c in enumerate(vocab)}
    ds = ManifestDataset(rows, labels, audio_conf, root=root)
    spect, target = ds[1]
    assert spect.shape[0] == 161 and spect.shape[1] > 0
    assert target == [labels[c] for c in "ba ab"]
    loader, _ = manifest_loader(ds, batch_size=2, shuffle=False)
    inputs, targets, pct, tsizes = next(iter(loader))
    assert inputs.shape[0] == 2 and inputs.shape[1] == 1 and inputs.shape[2] == 161
    assert int(tsizes.sum()) == len(targets)


def test_sha256_is_verified_and_corruption_is_refused(shard_root, audio_conf):
    import hashlib

    root, rows = shard_root
    with open(root / "audio" / "train-000.tar", "rb") as fh:
        fh.seek(rows[0]["offset"])
        blob_bytes = fh.read(rows[0]["size"])
    rows[0]["sha256"] = hashlib.sha256(blob_bytes).hexdigest()
    ds = ManifestDataset(rows, {c: i for i, c in enumerate("_ ab")}, audio_conf, root=root)
    spect, _ = ds[0]  # intact: loads
    assert spect.shape[1] > 0
    shard = root / "audio" / "train-000.tar"
    data = bytearray(shard.read_bytes())
    data[rows[0]["offset"] + 100] ^= 0xFF  # flip one byte inside the clip
    shard.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="do not match"):
        _ = ds[0]
    unverified = ManifestDataset(
        rows, {c: i for i, c in enumerate("_ ab")}, audio_conf, root=root, verify=False
    )
    spect, _ = unverified[0]  # verification off: the flipped byte is not noticed
    assert spect.shape[1] > 0
