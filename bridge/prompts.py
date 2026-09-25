"""Prompt templates used by every evaluation, probe and steering run.

All prompts are wrapped in the backbone's chat template at generation time
(see ``bridge.model.generate``), so the templates below are the content of a
single user turn.
"""

# Bare prompt (q, no context): stable retention (SR, SDR) and the vacuum
# phase of the volatile test (VRR).
BARE_QA = (
    "Answer the following question concisely.\n\n"
    "Question: {question}\n"
    "Answer:"
)

# Bare prompt with an abstention instruction ("+ instruction" row).
BARE_QA_WITH_REFUSAL = (
    "Answer the following question concisely. "
    "If you are not sure or the answer may be outdated, say \"I don't know\".\n\n"
    "Question: {question}\n"
    "Answer:"
)

# Grounded prompt (c, q): contextual obedience (CO).
RAG_QA = (
    "Answer the following question using ONLY the provided context. "
    "If the context contradicts what you already know, trust the context.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

# RAG_QA with the same abstention sentence as BARE_QA_WITH_REFUSAL, so the
# "+ instruction" row reports CO under the instruction it reports SR/VRR under.
RAG_QA_WITH_REFUSAL = (
    "Answer the following question using ONLY the provided context. "
    "If the context contradicts what you already know, trust the context. "
    "If you are not sure or the answer may be outdated, say \"I don't know\".\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

# Grounded phase of the volatile test (reported in the appendix only).
RAG_QA_GROUNDED = (
    "Answer the following question using ONLY the provided context.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

# The deferral string d that A-VMD and CA-AVMD are trained to emit.
DEFERRAL_STRING = "I don't have reliable information about this."
