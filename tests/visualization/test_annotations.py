"""Tests for locating entity spans in texts (displaCy visualization)."""

import pytest

from mdner_llm.models.entities import ListOfEntities
from mdner_llm.visualization.annotations import (
    convert_annotations_from_model,
    find_free_spans,
)


def find(text: str, span: str, *, find_all: bool = True) -> list[str]:
    """Return the texts of the spans found (helper to keep tests readable).

    Returns
    -------
    list[str]
        The matched substrings of the original text.
    """
    consumed = [False] * len(text)
    spans = find_free_spans(text.lower(), span, consumed, find_all=find_all)
    return [text[start:end] for start, end in spans]


@pytest.mark.parametrize(
    ("text", "span", "expected"),
    [
        # Whole words are found, whatever the surrounding punctuation.
        ("[408 POPC, 35840 SOL]", "SOL", ["SOL"]),
        ("POPC/PSM/Ergo (2:2:1)", "Ergo", ["Ergo"]),
        ("Conditions: T=295K.", "295K", ["295K"]),
        # Partial words are not matched.
        ("A solution of water", "SOL", []),
        ("Ergosterol parameters", "Ergo", []),
        ("CHARMM36m force field", "CHARMM36", []),
        # Matching is case-insensitive and returns the original case.
        ("Gromacs and GROMACS", "gromacs", ["Gromacs", "GROMACS"]),
        # Regex special characters in the entity are matched literally.
        ("GROMACS 2019.6 was used", "2019.6", ["2019.6"]),
        ("Version 2019x6", "2019.6", []),
    ],
)
def test_find_free_spans_matches_whole_words_only(
    text: str, span: str, expected: list[str]
) -> None:
    """Only whole-word occurrences of the entity are located."""
    assert find(text, span) == expected


def test_find_free_spans_returns_first_occurrence_by_default() -> None:
    """Without find_all, only the first occurrence is returned."""
    assert find("POPC and POPC", "POPC", find_all=False) == ["POPC"]


def test_find_free_spans_skips_consumed_positions() -> None:
    """Occurrences overlapping an already placed span are skipped."""
    text = "Ergo and Ergo"
    consumed = [True] * 4 + [False] * (len(text) - 4)  # First 'Ergo' taken.

    spans = find_free_spans(text.lower(), "Ergo", consumed, find_all=True)

    assert spans == [(9, 13)]
    assert all(consumed[9:13])


def test_convert_annotations_places_longest_entities_first() -> None:
    """A short entity does not take a part of a longer one."""
    text = "Ternary mixture of POPC/PSM/Ergo with Ergo parameters"
    predictions = ListOfEntities.model_validate(
        {
            "entities": [
                {"category": "MOL", "text": "Ergo"},
                {"category": "MOL", "text": "PSM/Ergo"},
            ]
        }
    )

    entities = convert_annotations_from_model(predictions, text, find_all=True)[0][
        "ents"
    ]

    assert [text[e["start"] : e["end"]] for e in entities] == ["PSM/Ergo", "Ergo"]
