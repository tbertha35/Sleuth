"""Chat formats: turning a conversation into the exact text a chat model was trained on.

Instruction-tuned models only behave well inside their chat format. Gemma 4 IT,
fed raw text, loses ~19 nats per token on the first few tokens (owner's run;
llama.cpp agrees), but answers correctly once the prompt is a user turn.

A model's GGUF carries its format as a Jinja template (tokenizer.chat_template).
We don't add a Jinja engine (a dependency); instead each format the owner uses
is written out here by hand from its template, for the plain case: system, user
and assistant messages, no tools, no images.

  gemma4  (from the gemma-4-12B-it GGUF's template)
      <bos><|turn>system\n{system}<turn|>\n          only if there is a system message
      <|turn>user\n{text}<turn|>\n                   (or thinking is on)
      <|turn>model\n{text}<turn|>\n
      generation prompt: <|turn>model\n<|channel>thought\n<channel|>
      The empty "thought" channel tells the model to answer without thinking.
      With thinking on, the system turn starts with <|think|>\n and the
      channel is left for the model to fill.

  chatml  (Qwen2/2.5/3; the common "ChatML" convention. Qwen3's own template
      also handles thinking, which this doesn't yet: Qwen3 will think out loud)
      <|im_start|>{role}\n{text}<|im_end|>\n ... generation prompt: <|im_start|>assistant\n

The rendered text doesn't include <bos>: our encoders leave that to the caller
(`tokenizer.start_id`), and `encode_chat` adds it.
"""

from __future__ import annotations


def chat_format(tokenizer) -> str | None:
    """'gemma4', 'chatml' or None (a base model, e.g. our own)."""
    special = getattr(tokenizer, "special_tokens", {}) or {}
    if "<|turn>" in special and "<turn|>" in special:
        return "gemma4"
    if "<|im_start|>" in special and "<|im_end|>" in special:
        return "chatml"
    return None


def render_chat(tokenizer, messages: list[dict], add_generation_prompt: bool = True, thinking: bool = False) -> str:
    """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]"""
    fmt = chat_format(tokenizer)
    if fmt == "gemma4":
        return _gemma4(messages, add_generation_prompt, thinking)
    if fmt == "chatml":
        return _chatml(messages, add_generation_prompt)
    raise ValueError("this model's tokenizer has no chat format we know (it's probably a base model; "
                     "use a plain prompt instead)")


def encode_chat(tokenizer, messages: list[dict], add_generation_prompt: bool = True, thinking: bool = False) -> list[int]:
    text = render_chat(tokenizer, messages, add_generation_prompt, thinking)
    start = [tokenizer.start_id] if tokenizer.start_id is not None else []
    return start + tokenizer.encode(text, allowed_special="all")


def _strip_thinking(text: str) -> str:
    """The template's strip_thinking: drop <|channel>...<channel|> blocks from model turns."""
    out = ""
    for part in text.split("<channel|>"):
        out += part.split("<|channel>")[0] if "<|channel>" in part else part
    return out.strip()


def _gemma4(messages, add_generation_prompt, thinking):
    out = ""
    rest = list(messages)
    if thinking or (rest and rest[0]["role"] in ("system", "developer")):
        out += "<|turn>system\n"
        if thinking:
            out += "<|think|>\n"
        if rest and rest[0]["role"] in ("system", "developer"):
            out += rest.pop(0)["content"].strip()
        out += "<turn|>\n"
    for i, m in enumerate(rest):
        role = "model" if m["role"] == "assistant" else m["role"]
        # Consecutive assistant messages continue one model turn (the template's
        # continue_same_model_turn); rare, but kept for fidelity.
        continuing = role == "model" and i > 0 and rest[i - 1]["role"] == "assistant"
        if not continuing:
            out += f"<|turn>{role}\n"
        out += _strip_thinking(m["content"]) if role == "model" else m["content"].strip()
        if not (role == "model" and i + 1 < len(rest) and rest[i + 1]["role"] == "assistant"):
            out += "<turn|>\n"
    if add_generation_prompt:
        out += "<|turn>model\n"
        if not thinking:
            out += "<|channel>thought\n<channel|>"
    return out


def _chatml(messages, add_generation_prompt):
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    if add_generation_prompt:
        out += "<|im_start|>assistant\n"
    return out
