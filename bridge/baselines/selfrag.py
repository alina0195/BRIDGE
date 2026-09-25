"""Self-RAG adaptive retrieval with vLLM.

Two-stage inference of Asai et al., "Self-RAG: Learning to Retrieve, Generate,
and Critique through Self-Reflection" (ICLR 2024; github.com/AkariAsai/self-rag),
as in the short-form adaptive-retrieval setting of the reference code:

    1. decode the bare prompt and read P(retrieve) from the first-token
       probabilities of the [Retrieval] / [No Retrieval] reflection tokens;
    2. if P(retrieve) > threshold and a passage is available, decode again
       with the passage; otherwise decode with [No Retrieval] appended.

``force_retrieve`` skips the decision and always decodes with the passage.
Requires a vLLM ``LLM`` and ``SamplingParams(logprobs=k, temperature=0,
skip_special_tokens=False)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CONTROL_TOKENS = [
    "[Fully supported]", "[Partially supported]",
    "[No support / Contradictory]",
    "[No Retrieval]", "[Retrieval]",
    "[Continue to Use Evidence]",
    "[Irrelevant]", "[Relevant]",
    "<paragraph>", "</paragraph>",
    "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]",
]

RETRIEVAL_TOKENS = ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"]


def format_prompt(instruction: str, paragraph: str | None = None) -> str:
    """Self-RAG prompt format (reference quick start)."""
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    if paragraph is not None:
        prompt += f"[Retrieval]<paragraph>{paragraph}</paragraph>"
    return prompt


def strip_special_tokens(text: str) -> str:
    for tok in CONTROL_TOKENS:
        text = text.replace(tok, "")
    return text.replace("</s>", "").replace("<s>", "").strip()


def p_retrieve(logprobs: list | None, ret_token_ids: dict[str, int]) -> float:
    """P([Retrieval]) / (P([Retrieval]) + P([No Retrieval])) at the first token.

    Tokens outside the returned top-k logprobs count as probability 0.
    """
    if not logprobs:
        return 0.0
    first = logprobs[0]
    ret_prob = no_ret_prob = 0.0
    for name, tok_id in ret_token_ids.items():
        if tok_id in first:
            p = float(np.exp(float(first[tok_id].logprob)))
            if name == "[Retrieval]":
                ret_prob = p
            elif name == "[No Retrieval]":
                no_ret_prob = p
    denom = ret_prob + no_ret_prob
    return ret_prob / denom if denom > 0 else 0.0


def selfrag_generate(
    model: Any,
    tokenizer: Any,
    instruction: str,
    paragraph: str | None = None,
    threshold: float = 0.5,
    sampling_params: Any = None,
    force_retrieve: bool = False,
) -> dict:
    """Answer one instruction; returns clean_answer, did_retrieve, p_retrieve, branch."""
    has_paragraph = paragraph is not None and str(paragraph).strip() != ""

    if force_retrieve:
        if not has_paragraph:
            raise ValueError("force_retrieve requires a non-empty paragraph")
        raw = model.generate([format_prompt(instruction, paragraph)],
                             sampling_params)[0].outputs[0].text
        return {"clean_answer": strip_special_tokens(raw), "did_retrieve": True,
                "p_retrieve": 1.0, "branch": "retrieval_forced"}

    ret_token_ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in RETRIEVAL_TOKENS}
    first_pass = model.generate([format_prompt(instruction)], sampling_params)[0]
    p = p_retrieve(first_pass.outputs[0].logprobs, ret_token_ids)

    if p > threshold and has_paragraph:
        prompt, branch = format_prompt(instruction, paragraph), "retrieval"
    else:
        prompt, branch = format_prompt(instruction) + "[No Retrieval]", "no_retrieval"
    raw = model.generate([prompt], sampling_params)[0].outputs[0].text
    return {"clean_answer": strip_special_tokens(raw), "did_retrieve": branch == "retrieval",
            "p_retrieve": p, "branch": branch}
