""" """

import operator

import pandas as pd
from loguru import logger


def compare_entities(groundtruth: dict, response: dict) -> None:
    """
    Compare groundtruth entities with predicted entities and print a summary table.

    Parameters
    ----------
        groundtruth (dict): Ground truth entities in format {"entities": [{"label":...,
        "text":..., "start":..., "end":...}, ...]}
        response (dict): Predicted entities in format {"entities": {"LABEL": [{"text":
        ..., "confidence":..., "start":..., "end":...}, ...], ...}}
    """
    gt_entities = groundtruth["entities"]
    pred_entities = response["entities"]

    # Flatten predicted entities into a list of dicts
    pred_list = [
        {"label": label, "text": ent["text"], "start": ent["start"], "end": ent["end"]}
        for label, ents in pred_entities.items()
        for ent in ents
    ]

    # Remove duplicates in groundtruth (non-redondant)
    seen = set()
    gt_unique = []
    for e in gt_entities:
        key = (e["label"], e["text"])
        if key not in seen:
            seen.add(key)
            gt_unique.append(e)

    # Sort by label and text
    gt_sorted = sorted(gt_unique, key=operator.itemgetter("label", "text"))
    pred_sorted = sorted(pred_list, key=operator.itemgetter("label", "text"))

    # Build table for groundtruth matches
    table_gt = [
        {
            "label_gt": gt["label"],
            "text_gt": gt["text"],
            "label_predicted": next(
                (
                    p["label"]
                    for p in pred_sorted
                    if p["label"] == gt["label"] and p["text"] == gt["text"]
                ),
                "",
            ),
            "text_predicted": next(
                (
                    p["text"]
                    for p in pred_sorted
                    if p["label"] == gt["label"] and p["text"] == gt["text"]
                ),
                "",
            ),
            "true_predicted": "✅"
            if any(
                gt["label"] == p["label"] and gt["text"] == p["text"]
                for p in pred_sorted
            )
            else "❌",
        }
        for gt in gt_sorted
    ]

    # Identify predictions not in groundtruth
    gt_keys = {(gt["label"], gt["text"]) for gt in gt_sorted}
    table_extra = [
        {
            "label_gt": "",
            "text_gt": "",
            "label_predicted": p["label"],
            "text_predicted": p["text"],
            "true_predicted": "❌",
        }
        for p in pred_sorted
        if (p["label"], p["text"]) not in gt_keys
    ]

    # Combine tables: groundtruth first, extras after
    table = table_gt + table_extra

    # Calculate statistics
    total_gt = len(gt_sorted)
    total_pred = len(pred_sorted)
    true_no_pos = sum(1 for row in table_gt if row["true_predicted"] == "✅")
    true_with_pos = sum(
        1
        for gt in gt_sorted
        if any(
            gt["label"] == p["label"]
            and gt["text"] == p["text"]
            and gt["start"] == p["start"]
            and gt["end"] == p["end"]
            for p in pred_sorted
        )
    )

    # Display table
    df = pd.DataFrame(
        table,
        columns=[
            "label_gt",
            "text_gt",
            "label_predicted",
            "text_predicted",
            "true_predicted",
        ],
    )
    print(df.to_markdown(index=False))

    # Display summary
    print("\n--- Summary ---")
    print(f"Total entities in groundtruth: {total_gt}")
    print(
        "True predicted / total predicted (ignoring positions): "
        f"{true_no_pos}/{total_pred} = {true_no_pos / total_pred:.2%}"
    )
    print(
        "True predicted / total predicted (with positions): "
        f"{true_with_pos}/{total_pred} = {true_with_pos / total_pred:.2%}"
    )


def _add_span(
    ents: list[dict[str, int | str]],
    consumed: list[bool],
    start: int,
    end: int,
    label: str,
) -> None:
    """Add a non-overlapping span."""
    if any(consumed[start:end]):
        return

    for i in range(start, end):
        consumed[i] = True

    ents.append({"start": start, "end": end, "label": label})


def _process_llm_entities(
    entities_data: dict[str, list[dict[str, dict]]],
    text: str,
) -> list[dict[str, int | str]]:
    """Handle nested LLM-style entity format.

    Parameters
    ----------
    entities_data : dict[str, list[dict[str, dict]]]
        Entity data in LLM format.
    text : str
        Text to search for entities.

    Returns
    -------
    list[dict[str, int | str]]
        List of entities with start, end, and label.
    """
    ents: list[dict[str, int | str]] = []
    consumed = [False] * len(text)
    text_lower = text.lower()

    for label, ent_list in entities_data.items():
        for ent in ent_list:
            span_text = ent["text"]
            span_lower = span_text.lower()
            found = False

            # Explicit positions first
            if "start" in ent and "end" in ent:
                _add_span(ents, consumed, ent["start"], ent["end"], label)
                found = True

            # Fallback search
            search_pos = 0
            while True:
                start = text_lower.find(span_lower, search_pos)
                if start == -1:
                    break
                end = start + len(span_text)

                if not any(consumed[start:end]):
                    _add_span(ents, consumed, start, end, label)
                    found = True

                search_pos = start + 1

            if not found:
                logger.warning(
                    "Entity '%s' with label '%s' not found.",
                    span_text,
                    label,
                )

    return ents


def _process_groundtruth_entities(
    entities_data: list[dict[str, dict]],
) -> list[dict[str, int | str]]:
    """Handle flat groundtruth entity format.

    Parameters
    ----------
    entities_data : list[dict[str, dict]]
        Entity data in groundtruth format.

    Returns
    -------
    list[dict[str, int | str]]
        List of entities with start, end, and label.
    """
    return [
        {"start": e["start"], "end": e["end"], "label": e["label"]}
        for e in entities_data
    ]


def convert_annotations_llm(
    response: dict[str, dict],
    text_to_annotate: str,
) -> list[dict[str, dict]]:
    """
    Convert LLM or groundtruth entities to spaCy displaCy format.

    Parameters
    ----------
    response : dict[str, dict]
        Entity response containing key "entities".
    text_to_annotate : str
        Original text.

    Returns
    -------
    list[dict[str, dict]]
        displaCy-compatible structure.

    Raises
    ------
    ValueError
        If response does not contain key "entities".
    TypeError
        If entities format is neither a dict nor a list.

    """
    if "entities" not in response:
        msg = "Response must contain key 'entities'."
        raise ValueError(msg)

    entities_data = response["entities"]

    if isinstance(entities_data, dict):
        ents = _process_llm_entities(entities_data, text_to_annotate)
    elif isinstance(entities_data, list):
        ents = _process_groundtruth_entities(entities_data)
    else:
        msg = "Unknown entities format."
        raise TypeError(msg)

    return [{"text": text_to_annotate.replace("\n", " "), "ents": ents}]
