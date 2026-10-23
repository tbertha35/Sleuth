"""Registry of analyzer views.

A view is a module with two functions:
    run(model, tokenizer, cases, device, **options) -> {"summary": dict, "arrays": {name: np.ndarray}}
    timeline_series(summary) -> {name: series(...)}   (common.series: one value per row)

`summary` is small JSON for tables and timelines; `arrays` holds the bigger
numeric data (per neuron, per head, attention patterns) saved as .npz.
To add a view: write the module, add it here, and the timeline and dashboard
pick it up.
"""

from . import ablation, attention, logit_lens, neurons

VIEWS = {
    "neuron_stats": neurons,
    "attention": attention,
    "logit_lens": logit_lens,
    "ablation": ablation,
}
