"""FLARE (Forward-Looking Active REtrieval) for local HuggingFace models.

Adapts the confidence-triggered retrieval loop of Jiang et al., "Active
Retrieval Augmented Generation" (EMNLP 2023; github.com/jzbjyb/FLARE):

    1. generate a look-ahead continuation with per-token probabilities;
    2. if any token probability is below ``filter_threshold`` and a passage
       is available, "retrieve" it, regenerate with the passage and stop;
    3. otherwise accept the first sentence of the look-ahead and continue,
       for at most ``max_iterations`` sentences.

Retrieval returns the item's gold passage (``golden_context``) instead of
querying a search index, so no masked query is formed.  With
``paragraph=None`` nothing can be retrieved, but the trigger decision is still
recorded (``trigger_fired``, ``p_min_lookahead``).
"""

from __future__ import annotations

from typing import Any

import torch
from nltk.tokenize import PunktSentenceTokenizer

from bridge.model import chat_input_ids, input_device

_punkt = PunktSentenceTokenizer()

# Passage-conditioned prompt used when the trigger fires (same wording as
# bridge.prompts.RAG_QA, followed by the BARE_QA instruction).
_RAG_PREFIX = (
    "Answer the following question using ONLY the provided context. "
    "If the context contradicts what you already know, trust the context.\n\n"
    "Context: {paragraph}\n\n"
)


def generate_with_logprobs(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int = 64,
    use_chat_template: bool = True,
) -> tuple[str, list[str], list[float]]:
    """Greedy generation; returns (text, token strings, token probabilities)."""
    device = input_device(model)
    if use_chat_template:
        input_ids = chat_input_ids(tokenizer, prompt).to(device)
    else:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = encoded["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    generated_ids = outputs.sequences[0][prompt_len:]
    tokens: list[str] = []
    probs: list[float] = []
    for i, token_id in enumerate(generated_ids):
        tid = token_id.item()
        if tid == tokenizer.eos_token_id:
            break
        tokens.append(tokenizer.decode([tid]))
        probs.append(torch.softmax(outputs.scores[i][0], dim=-1)[tid].item())
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    del input_ids, outputs, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text, tokens, probs


def _first_sentence(text: str) -> str:
    spans = list(_punkt.span_tokenize(text))
    if not spans:
        return text
    start, end = spans[0]
    return text[start:end].rstrip()


def flare_generate(
    model: Any,
    tokenizer: Any,
    instruction: str,
    paragraph: str | None = None,
    filter_threshold: float = 0.5,
    max_iterations: int = 5,
    max_new_tokens: int = 64,
) -> dict:
    """Run the FLARE loop on one instruction.

    Returns clean_answer, did_retrieve, retrieval_count, p_min_token (lowest
    token probability of the final generation), trigger_fired and
    p_min_lookahead (the trigger decision and the lowest look-ahead token
    probability, recorded whether or not a passage was available).
    """
    retrieval_count = 0
    did_retrieve = False
    trigger_fired = False
    p_min_lookahead = 1.0
    accumulated = ""
    probs: list[float] = []

    for _ in range(max_iterations):
        prompt = instruction + " " + accumulated if accumulated else instruction
        # Continuation steps feed the raw prompt, without the chat template.
        text, tokens, probs = generate_with_logprobs(
            prompt, tokenizer, model, max_new_tokens=max_new_tokens,
            use_chat_template=not accumulated,
        )
        if not tokens:
            break

        p_min_lookahead = min(p_min_lookahead, min(probs) if probs else 1.0)
        low_conf = any(p < filter_threshold for p in probs)
        trigger_fired = trigger_fired or low_conf

        if low_conf and paragraph and paragraph.strip():
            retrieval_count += 1
            did_retrieve = True
            rag_prompt = _RAG_PREFIX.format(paragraph=paragraph) + instruction
            if accumulated:
                rag_prompt = rag_prompt + " " + accumulated
            text, tokens, probs = generate_with_logprobs(
                rag_prompt, tokenizer, model, max_new_tokens=max_new_tokens,
                use_chat_template=not accumulated,
            )
            accumulated = (accumulated + " " + text).strip() if accumulated else text
            break

        first = _first_sentence(text)
        accumulated = (accumulated + " " + first).strip() if accumulated else first
        if first == text:
            break

    return {
        "clean_answer": accumulated.strip(),
        "did_retrieve": did_retrieve,
        "retrieval_count": retrieval_count,
        "p_min_token": min(probs) if probs else 1.0,
        "trigger_fired": trigger_fired,
        "p_min_lookahead": p_min_lookahead,
    }
