from sleuth.cli import main


def test_info(capsys):
    assert main(["info", "configs/tiny-10m.toml"]) == 0
    out = capsys.readouterr().out
    assert "non-embedding" in out and "FLOPs" in out


def test_tokenize(tiny_data, capsys):
    assert main(["tokenize", "--tokenizer", str(tiny_data["tokenizer"]), "Once upon a time"]) == 0
    assert "tokens" in capsys.readouterr().out


def test_score_reports_answer_loss_with_error_bars(tiny_data, tmp_path, capsys):
    import json

    from helpers import tiny_run_config

    from sleuth.forge.train.trainer import train

    run = train(tiny_run_config(tiny_data, tmp_path / "runs"), log=lambda s: None)
    out = tmp_path / "score.json"
    assert main(["score", str(run), "--suite", "suites/knowledge", "--device", "cpu", "--show-items",
                 "--json", str(out)]) == 0
    text = capsys.readouterr().out
    header = next(line for line in text.splitlines() if line.startswith("category"))
    assert header.index("answer loss") < header.index("exact match") < header.index("loss ", header.index("exact"))
    assert "±" in text and "facts-00" in text
    r = json.loads(out.read_text(encoding="utf-8"))
    assert len(r["cases"]) >= 200 and r["categories"]["overall"]["n_answers"] == len(r["cases"])
    assert 0 < r["standard_errors"]["overall"]["answer_loss"] < 1
    assert all("top1_text" in c for c in r["cases"])


def test_standard_errors():
    from sleuth.analyzer.common import standard_errors

    rows = [{"category": "a", "answer_loss": x, "exact": e} for x, e in [(1.0, True), (3.0, False)]]
    se = standard_errors(rows)
    assert abs(se["a"]["answer_loss"] - 1.0) < 1e-9          # std 1.414 / sqrt(2)
    assert abs(se["overall"]["exact_match"] - 0.5 / 2 ** 0.5) < 1e-9
