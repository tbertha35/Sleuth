"""Document masking (M11a step 1): attention stops at <|bos|>."""

import json

import pytest
import torch

from sleuth.forge.data.loader import document_ids
from sleuth.forge.infer.kv_cache import KVCache
from sleuth.forge.model.attention import doc_block_mask
from sleuth.forge.model.config import ModelConfig
from sleuth.forge.model.taps import listen
from sleuth.forge.model.transformer import Transformer
from sleuth.forge.train.trainer import train

from helpers import tiny_run_config

BOS = 1


def test_document_ids_count_bos_tokens():
    x = torch.tensor([[7, 8, BOS, 9, 9, BOS, 5],
                      [BOS, 4, 4, 4, 4, 4, 4]])
    assert document_ids(x, BOS).tolist() == [[0, 0, 1, 1, 1, 2, 2],
                                             [1, 1, 1, 1, 1, 1, 1]]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Transformer(ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                                   max_seq_len=64)).eval()


def _two_docs():
    torch.manual_seed(1)
    a = torch.randint(2, 64, (1, 9))
    b = torch.randint(2, 64, (1, 12))
    b[0, 0] = BOS
    return a, b, torch.cat([a, b], dim=1)


@pytest.mark.parametrize("explicit", [False, True])
def test_a_masked_document_sees_only_itself(model, explicit):
    """The second document in a masked window gives the same logits as on its
    own: no information leaks from the first. Positions keep counting across
    the boundary, which is fine because RoPE only sees query-key distances.
    `explicit` runs the written-out attention the analyzer uses."""
    a, b, both = _two_docs()
    # A listener on the attention pattern switches every layer to the explicit path.
    taps = {"blocks.*.attn.pattern": lambda name, t: None} if explicit else {}
    with torch.no_grad(), listen(model, taps):
        alone = model(b).logits
        masked = model(both, doc_ids=document_ids(both, BOS)).logits
        unmasked = model(both).logits
    n = a.shape[1]
    torch.testing.assert_close(masked[:, n:], alone, atol=1e-5, rtol=1e-5)
    assert (unmasked[:, n:] - alone).abs().max() > 1e-3   # without the mask, document 1 is read
    torch.testing.assert_close(masked[:, :n], unmasked[:, :n])  # the first document can't see ahead anyway


def test_one_document_is_the_same_as_no_mask(model):
    _, b, _ = _two_docs()
    with torch.no_grad():
        torch.testing.assert_close(model(b, doc_ids=document_ids(b, BOS)).logits, model(b).logits)


def test_doc_ids_and_a_cache_dont_mix(model):
    _, b, _ = _two_docs()
    cache = KVCache.for_model(model, 1, 32)
    with pytest.raises(ValueError, match="doc_ids"):
        model(b, cache=cache, doc_ids=document_ids(b, BOS))


def test_training_with_doc_mask(tiny_data, tmp_path):
    lines = []
    run = train(tiny_run_config(tiny_data, tmp_path, max_steps=6, lr=3e-3, doc_mask=True), log=lines.append)
    assert any("document-masked attention" in line for line in lines)
    recs = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in recs if "loss" in r)
    assert any("val_loss" in r for r in recs)


def test_flex_attention_gives_the_same_result(model):
    """The FlexAttention path (train.doc_mask_kernel = "flex") computes what the
    SDPA mask does. On CPU without compile it's FlexAttention's reference version."""
    a, b, both = _two_docs()
    doc = document_ids(both, BOS)
    with torch.no_grad(), pytest.warns(UserWarning, match="without torch.compile"):
        flex = model(both, block_mask=doc_block_mask(doc)).logits
    with torch.no_grad():
        sdpa = model(both, doc_ids=doc).logits
        alone = model(b).logits
    torch.testing.assert_close(flex, sdpa, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(flex[:, a.shape[1]:], alone, atol=1e-5, rtol=1e-5)


def test_the_flex_kernel_is_checked_before_training(tiny_data, tmp_path):
    """Training with it runs on the GPU only (test_gpu.py): FlexAttention has no
    backward pass on CPU, so the trainer says so instead of failing mid-step."""
    with pytest.raises(ValueError, match="needs a GPU"):
        train(tiny_run_config(tiny_data, tmp_path / "cpu", doc_mask=True, doc_mask_kernel="flex"), log=quiet)
    with pytest.raises(ValueError, match="doc_mask_kernel"):
        train(tiny_run_config(tiny_data, tmp_path / "bad", doc_mask=True, doc_mask_kernel="fast"), log=quiet)
    assert not (tmp_path / "cpu").exists() and not (tmp_path / "bad").exists()   # no half-made run folders


def quiet(_):
    pass
