from datasets import load_dataset, Dataset

from data.format.prompts import PROMPT, CODE_PROMPT
from data.format.utils import cast_large_strings


def _add_prompt(ex):
    if ex["kind"] == "code":
        return CODE_PROMPT.format(problem=ex["problem"])
    else:
        return PROMPT.format(problem=ex["problem"])


def load_train(category: str = None, dataset_filter: str = None) -> Dataset:
    """Load lasgroup/verifiable-corpus. Filter by kind (category) and/or by dataset name (e.g. dapo_math)."""
    ds = load_dataset("lasgroup/verifiable-corpus", split="train")
    ds = cast_large_strings(ds, columns=list(ds.features.keys()))
    if category is not None:
        ds = ds.filter(lambda ex: ex["kind"] == category)
    if dataset_filter is not None:
        ds = ds.filter(lambda ex: ex.get("dataset") == dataset_filter)
    return ds.map(lambda ex: {"prompt": _add_prompt(ex)})
