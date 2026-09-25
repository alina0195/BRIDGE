"""Shared colours and markers for the figures.

Backbone colours form a grey ramp used when the backbone is the only
identity axis in a figure; the marker repeats the identity for greyscale.
"""

BACKBONE_LABELS = {
    "llama": "Llama-3.1-8B",
    "mistral": "Mistral-7B",
    "qwen": "Qwen2.5-7B",
}

BACKBONE_COLORS = {
    "llama": "#5c5a53",
    "mistral": "#87857c",
    "qwen": "#b0aea3",
}

BACKBONE_MARKERS = {
    "llama": "o",
    "mistral": "s",
    "qwen": "^",
}
