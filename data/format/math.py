from datasets import load_dataset, Dataset, concatenate_datasets

from data.format.prompts import PROMPT
from data.utils.math import process_gsm8k

PROBLEM_KEY = {
    "math-ai/aime25": "problem",
    "math-ai/aime24":"problem",
    "math-ai/amc23":"question",
    "math-ai/math500":"problem",
    "openai/gsm8k": "question",
    "EleutherAI/hendrycks_math": "problem",
}
ANSWER_KEY = {
    "math-ai/aime25": "answer",
    "math-ai/aime24":"solution",
    "math-ai/amc23":"answer",
    "math-ai/math500":"answer",
    "openai/gsm8k": "answer",
    "EleutherAI/hendrycks_math": "answer",
}


def _format_math(ex, dataset_name: str) -> dict:
    return {
        "kind": "math",
        "dataset": dataset_name.split("/")[1],
        "description": ex[PROBLEM_KEY[dataset_name]],
        "problem": ex[PROBLEM_KEY[dataset_name]],
        "prompt": PROMPT.format(problem=ex[PROBLEM_KEY[dataset_name]]),
        "answer": str(ex[ANSWER_KEY[dataset_name]]),
    }


def load_math(dataset_name: str, split: str | None = None) -> Dataset:
    assert dataset_name in [
        "math-ai/aime24",
        "math-ai/aime25",
        "math-ai/math500",
        "math-ai/amc23",
        "openai/gsm8k",
        "EleutherAI/hendrycks_math",
    ]

    if dataset_name == "openai/gsm8k":
        # Use requested split (e.g. "train" or "test" for original HF splits); default "test" for backward compatibility
        hf_split = split if split is not None else "test"
        ds = load_dataset("openai/gsm8k", "main", split=hf_split)
    elif dataset_name == "EleutherAI/hendrycks_math":
        # Use all subject subsets (configs) and both train/test splits
        hendrycks_subjects = [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]

        all_splits = []
        for subject in hendrycks_subjects:
            all_splits.append(load_dataset(dataset_name, subject, split="train"))
            all_splits.append(load_dataset(dataset_name, subject, split="test"))

        ds = concatenate_datasets(all_splits)

        # Extract a short final answer from the full worked solution.
        # Reuse the normalization utilities from math_dapo (adapted from Hendrycks MATH utils).
        from verl.utils.reward_score import math_dapo as _math_dapo_utils

        def _extract_final_answer(ex):
            solution = ex.get("solution", "")
            boxed = _math_dapo_utils.last_boxed_only_string(solution)
            if boxed is None:
                return {"answer": ""}
            try:
                ans = _math_dapo_utils.normalize_final_answer(
                    _math_dapo_utils.remove_boxed(boxed)
                )
            except Exception:
                ans = ""
            return {"answer": ans}

        ds = ds.map(_extract_final_answer, desc="Extract final answer from Hendrycks MATH solution")
    else:
        ds = load_dataset(dataset_name, split="test")

    if dataset_name == "math-ai/aime24":  # remove \boxed{}
        ds = ds.map(lambda ex: {ANSWER_KEY[dataset_name]: ex[ANSWER_KEY[dataset_name]][7:-1]}, desc="AIME24 answer extraction")
    if dataset_name == "openai/gsm8k":
        ds = ds.map(process_gsm8k, desc="GSM8K answer extraction")

    return ds.map(lambda ex: _format_math(ex, dataset_name=dataset_name), remove_columns=ds.column_names)
