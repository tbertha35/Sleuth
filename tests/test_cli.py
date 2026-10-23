from sleuth.cli import main


def test_info(capsys):
    assert main(["info", "configs/tiny-10m.toml"]) == 0
    out = capsys.readouterr().out
    assert "non-embedding" in out and "FLOPs" in out


def test_tokenize(tiny_data, capsys):
    assert main(["tokenize", "--tokenizer", str(tiny_data["tokenizer"]), "Once upon a time"]) == 0
    assert "tokens" in capsys.readouterr().out
