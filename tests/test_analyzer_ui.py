import json
import threading
import urllib.request
from urllib.error import HTTPError

import numpy as np
import pytest
import torch
from helpers import tiny_run_config

from sleuth.analyzer.neurons import neuron_stats
from sleuth.analyzer.suites import load_suite
from sleuth.analyzer.timeline import run_timeline
from sleuth.forge.model.config import ModelConfig
from sleuth.forge.model.transformer import Transformer
from sleuth.forge.tokenizer.bpe import Tokenizer
from sleuth.forge.train.checkpoint import list_snapshots
from sleuth.forge.train.trainer import train
from sleuth.ui.server import make_server

SUITE = "suites/basic"


@pytest.fixture(scope="module")
def analyzed_run(tiny_data, tmp_path_factory):
    run = train(tiny_run_config(tiny_data, tmp_path_factory.mktemp("runs")), log=lambda s: None)
    run_timeline(run, SUITE, log=lambda s: None)
    return run


def test_suite_loads_with_categories():
    cases = load_suite(SUITE)
    assert {c.category for c in cases} == {"story", "math", "code"}
    assert any(c.expected for c in cases)


def test_neuron_stats_shapes_and_ranges(tiny_data):
    tok = Tokenizer.load(tiny_data["tokenizer"])
    torch.manual_seed(0)
    model = Transformer(ModelConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=2, n_heads=4, max_seq_len=128))
    res = neuron_stats(model, tok, load_suite(SUITE))
    a, s = res["arrays"], res["summary"]
    H = model.cfg.hidden
    assert a["fire_rate"].shape == (2, H) and a["fire_rate_by_category"].shape == (2, 3, H)
    assert np.all((a["fire_rate"] >= 0) & (a["fire_rate"] <= 1))
    # The overall rate is the token-weighted mix of the per-category rates.
    w = np.array([s["tokens_by_category"][c] for c in s["categories"]], dtype=np.float64)
    mixed = (a["fire_rate_by_category"] * w[None, :, None]).sum(1) / w.sum()
    np.testing.assert_allclose(mixed, a["fire_rate"], atol=1e-5)
    for lay in s["layers"]:
        assert 0 <= lay["frac_selective"] <= 1 and lay["act_rms"] > 0


def test_timeline_covers_every_snapshot(analyzed_run):
    root = analyzed_run / "analysis" / "neuron_stats" / "basic"
    t = json.loads((root / "timeline.json").read_text())
    assert t["steps"] == [s for s, _ in list_snapshots(analyzed_run)]
    assert len(t["metrics"]["frac_selective"]) == len(t["steps"])
    assert len(t["metrics"]["frac_selective"][0]) == t["n_layers"]
    # Re-running skips finished checkpoints (so it's cheap to call during training).
    mtime = (root / "step_0000000" / "summary.json").stat().st_mtime
    run_timeline(analyzed_run, SUITE, log=lambda s: None)
    assert (root / "step_0000000" / "summary.json").stat().st_mtime == mtime


@pytest.fixture(scope="module")
def server(analyzed_run):
    httpd = make_server(analyzed_run.parent, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read()


def test_api_endpoints(server, analyzed_run):
    rid = analyzed_run.name
    runs = json.loads(get(f"{server}/api/runs")[1])
    assert [r["id"] for r in runs] == [rid]
    detail = json.loads(get(f"{server}/api/runs/{rid}")[1])
    assert detail["analyses"][0]["suite"] == "basic" and detail["params"]["total"] > 0
    metrics = json.loads(get(f"{server}/api/runs/{rid}/metrics")[1])
    assert metrics["train"] and metrics["val"]
    tl = json.loads(get(f"{server}/api/runs/{rid}/analysis/neuron_stats/basic/timeline")[1])
    ck = json.loads(get(f"{server}/api/runs/{rid}/analysis/neuron_stats/basic/step_{tl['steps'][-1]:07d}")[1])
    assert len(ck["arrays"]["fire_rate"]) == tl["n_layers"]


def test_static_and_path_traversal(server):
    status, body = get(f"{server}/")
    assert status == 200 and b"<title>Sleuth</title>" in body
    for bad in ("/api/runs/..%2F..%2Fetc", "/../../etc/passwd",
                "/api/runs/x/analysis/a/..%2F..%2F/timeline", "/nope.js"):
        with pytest.raises(HTTPError) as e:
            get(server + bad)
        assert e.value.code in (400, 404)
