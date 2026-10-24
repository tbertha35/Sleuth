import json

import numpy as np
import pytest

from sleuth.forge.data.ingest import find_files, is_validation, iter_documents
from sleuth.forge.data.loader import ShardDataset
from sleuth.forge.data.shard import build_shards
from sleuth.forge.tokenizer.bpe import Tokenizer


def test_txt_separator_and_jsonl(tmp_path):
    # write_bytes, not write_text: on Windows write_text would turn every "\n"
    # into "\r\n" (so "\r\n" becomes "\r\r\n") and the test would no longer
    # be checking what it claims to check.
    (tmp_path / "a.txt").write_bytes(b"first doc\nline two\n<|endoftext|>\nsecond doc\n<|endoftext|>\n")
    (tmp_path / "b.txt").write_bytes(b"whole file\r\nis one doc\r\n")
    (tmp_path / "c.jsonl").write_text(json.dumps({"text": "j1", "category": "x"}) + "\n\n"
                                      + json.dumps({"other": 1}) + "\n" + json.dumps({"text": "j2"}) + "\n")
    (tmp_path / "_meta.txt").write_text("skipped")
    (tmp_path / "ignored.csv").write_text("skipped")
    files = find_files([tmp_path])
    assert [f.name for f in files] == ["a.txt", "b.txt", "c.jsonl"]
    docs = list(iter_documents(files))
    assert [d.text for d in docs] == ["first doc\nline two", "second doc", "whole file\nis one doc", "j1", "j2"]
    assert docs[3].category == "x"


def test_bad_jsonl_reports_line(tmp_path):
    (tmp_path / "x.jsonl").write_text('{"text": "ok"}\n{broken\n')
    with pytest.raises(ValueError, match="x.jsonl:2"):
        list(iter_documents([tmp_path / "x.jsonl"]))


def test_validation_split_is_stable():
    texts = [f"document number {i}" for i in range(2000)]
    flags = [is_validation(t, 0.1) for t in texts]
    assert flags == [is_validation(t, 0.1) for t in texts]
    assert 100 < sum(flags) < 300
    assert not any(is_validation(t, 0.0) for t in texts)


def test_shards_decode_back_to_documents(tiny_data):
    meta = json.loads((tiny_data["shards"] / "meta.json").read_text())
    tok = Tokenizer.load(tiny_data["tokenizer"])
    assert meta["dtype"] == "uint16" and meta["vocab_size"] == tok.vocab_size
    assert len(meta["splits"]["train"]) > 1  # small shard_tokens forced several files
    all_docs = [d.text for d in iter_documents(find_files([tiny_data["raw"]]))]
    train_docs = [t for t in all_docs if not is_validation(t, 0.05)]
    # First shard: documents appear in input order, each as [bos] ids [eos].
    first = meta["splits"]["train"][0]
    arr = np.fromfile(tiny_data["shards"] / first["file"], dtype=np.uint16)
    starts = np.load(tiny_data["shards"] / first["file"].replace(".bin", ".idx.npy"))
    assert len(arr) == first["n_tokens"] and len(starts) == first["n_docs"]
    for i in range(5):
        end = starts[i + 1]
        seg = arr[starts[i]:end].tolist()
        assert seg[0] == tok.bos_id and seg[-1] == tok.eos_id
        assert tok.decode(seg[1:-1]) == train_docs[i]


def test_shard_output_dir_must_be_empty(tiny_data):
    with pytest.raises(FileExistsError):
        build_shards([tiny_data["raw"]], tiny_data["tokenizer"], tiny_data["shards"], log=lambda s: None)


def test_parallel_sharding_is_identical(tiny_data, tmp_path):
    build_shards([tiny_data["raw"]], tiny_data["tokenizer"], tmp_path / "par", val_fraction=0.05,
                 shard_tokens=50_000, workers=2, log=lambda s: None)
    for f in sorted((tiny_data["shards"]).glob("*.bin")):
        assert (tmp_path / "par" / f.name).read_bytes() == f.read_bytes()


def test_loader_is_deterministic_and_shifted(tiny_data):
    ds = ShardDataset(tiny_data["shards"], "train")
    x1, y1 = ds.batch(8, 16, seed=1, step=5)
    x2, y2 = ds.batch(8, 16, seed=1, step=5)
    x3, _ = ds.batch(8, 16, seed=1, step=6)
    assert x1.shape == (8, 16) and x1.dtype.is_floating_point is False
    assert (x1 == x2).all() and (y1 == y2).all()
    assert not (x1 == x3).all()
    assert (x1[:, 1:] == y1[:, :-1]).all()  # y is x shifted by one


def test_sequential_validation_batches(tiny_data):
    ds = ShardDataset(tiny_data["shards"], "val")
    a = list(ds.sequential_batches(4, 16, max_batches=3))
    b = list(ds.sequential_batches(4, 16, max_batches=3))
    assert len(a) == 3
    assert all((xa == xb).all() for (xa, _), (xb, _) in zip(a, b))


def _write_parquet(path, texts, column="text"):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    pq.write_table(pa.table({column: texts, "id": list(range(len(texts)))}), path)


def test_parquet_and_size_caps(tmp_path):
    from sleuth.forge.data.ingest import iter_input_documents, parse_input

    (tmp_path / "web").mkdir()
    (tmp_path / "math").mkdir()
    _write_parquet(tmp_path / "web" / "000.parquet", [f"web document number {i}. " * 10 for i in range(50)])
    _write_parquet(tmp_path / "math" / "a.parquet", [f"{i} + {i} = {2 * i}" for i in range(20)])
    assert parse_input("data/raw/wiki@2.5") == (parse_input("data/raw/wiki")[0], 2_500_000)
    docs = list(iter_input_documents([tmp_path / "web", tmp_path / "math"]))
    assert len(docs) == 70 and {d.category for d in docs} == {"web", "math"}
    # A cap of 0.001 MB = 1000 bytes: only the first few web documents.
    capped = list(iter_input_documents([f"{tmp_path / 'web'}@0.001", tmp_path / "math"]))
    web = [d for d in capped if d.category == "web"]
    assert 0 < len(web) < 50 and sum(len(d.text) for d in web) <= 1000
    assert sum(d.category == "math" for d in capped) == 20
    with pytest.raises(ValueError, match="no column"):
        list(iter_documents([tmp_path / "web" / "000.parquet"], text_field="content"))


def test_shard_reports_the_mix(tiny_data, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "x.jsonl").write_bytes(b"".join(json.dumps({"text": f"Tom saw a cat {i}."}).encode() + b"\n"
                                                     for i in range(30)))
    (tmp_path / "b" / "y.txt").write_bytes(b"def f(x):\n    return x\n")
    meta = build_shards([tmp_path / "a", f"{tmp_path / 'b'}@1"], tiny_data["tokenizer"], tmp_path / "out",
                        val_fraction=0.0, log=lambda s: None)
    assert set(meta["category_bytes"]) == {"a", "b"} and meta["categories"]["a"] == 30
