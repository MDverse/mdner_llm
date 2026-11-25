"""Tests for the MetaMD Streamlit application.

External dependencies (GLiNER model, OpenRouter API, normalization databases
and web services, log file) are replaced by fakes: the tests run offline,
fast and deterministically.
"""

import json
from typing import Any

import httpx
import loguru
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from mdner_llm.gliner import extract_entities_with_gliner_all_texts as gliner_module
from mdner_llm.normalization import normalize_entities as normalization_module

# Relative to this test file (AppTest.from_file resolves it from here).
APP_PATH = "../streamlit_app.py"
TEXT = "A 100 ns simulation of POPC with GROMACS 2021 at 310 K."
# Entities returned by the fake GLiNER model, by category.
GLINER_PREDICTIONS = {
    "entities": {
        "MOL": [{"text": "POPC"}],
        "SOFTNAME": [{"text": "GROMACS"}],
        "SOFTVERS": [{"text": "2021"}],
        "STIME": [{"text": "100 ns"}],
        "STEMP": [{"text": "310 K"}],
    }
}


def fake_load_model(model_path: Any, adapter_path: Any, device: str) -> str:
    """Return a dummy model instead of loading GLiNER.

    Returns
    -------
    str
        A dummy model.
    """
    return "fake-model"


def fake_run_gliner(model: Any, text: str, entity_desc: dict) -> tuple:
    """Return fixed predictions instead of running GLiNER.

    Returns
    -------
    tuple
        Predictions, inference time, input and output token counts.
    """
    return GLINER_PREDICTIONS, 0.5, 10, 10


def fake_normalize_json_content(data: dict, **kwargs: Any) -> dict:
    """Keep every entity as is instead of querying databases and LLMs.

    Returns
    -------
    dict
        The data with the (unchanged) entities as normalized entities.
    """
    entities = [
        {**entity.model_dump(), "text_normalized": entity.text.lower()}
        for entity in data["formatted_response"].entities
    ]
    return {**data, "normalized_entities": {"entities": entities}}


def fake_quota_request(*args: Any, **kwargs: Any) -> httpx.Response:
    """Return a daily quota of 4 credits out of 5 instead of calling OpenRouter.

    Returns
    -------
    httpx.Response
        The fake OpenRouter response.
    """
    request = httpx.Request("GET", "https://openrouter.ai/api/v1/key")
    data = {"limit": 5, "limit_remaining": 4, "limit_reset": "daily"}
    return httpx.Response(200, json={"data": data}, request=request)


@pytest.fixture(autouse=True)
def offline_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the app in GLiNER mode with all external dependencies faked."""
    monkeypatch.setenv("DEFAULT_MODEL", "gliner")
    monkeypatch.setenv("OPENROUTER_STREAMLIT_API_KEY", "fake-key")
    monkeypatch.setattr(gliner_module, "load_model", fake_load_model)
    monkeypatch.setattr(gliner_module, "run_gliner", fake_run_gliner)
    monkeypatch.setattr(
        normalization_module, "normalize_json_content", fake_normalize_json_content
    )
    monkeypatch.setattr(httpx, "get", fake_quota_request)
    # Log to the console only (no log file written by the tests).
    monkeypatch.setattr("mdner_llm.logger.create_logger", lambda *_: loguru.logger)
    # Settings and model are cached across runs: start each test from scratch.
    st.cache_resource.clear()
    st.cache_data.clear()


def run_extraction(text: str = TEXT) -> AppTest:
    """Type a description, click on 'Extract Metadata' and wait for the results.

    Returns
    -------
    AppTest
        The app after the extraction and normalization steps.
    """
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    app.text_area(key="input_text").input(text).run()
    app.button[-1].click().run()  # 'Extract Metadata' is the last button.
    app.run()  # Normalization step.
    return app


@pytest.fixture
def exported_json(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the JSON data given to the export (download) button.

    Returns
    -------
    list[str]
        The exported JSON strings, one per rendering of the button.
    """
    exports = []
    original_download_button = st.download_button

    def recording_download_button(*args: Any, **kwargs: Any) -> bool:
        exports.append(kwargs["data"])
        return original_download_button(*args, **kwargs)

    monkeypatch.setattr(st, "download_button", recording_download_button)
    return exports


def test_export_maps_each_entity_text_to_its_category(
    exported_json: list[str],
) -> None:
    """The exported annotation contains the text and its entities."""
    run_extraction()

    export = json.loads(exported_json[-1])

    assert export == [
        {
            "text": TEXT,
            "entities": [
                {"POPC": "MOL"},
                {"GROMACS": "SOFTNAME"},
                {"2021": "SOFTVERS"},
                {"100 ns": "STIME"},
                {"310 K": "STEMP"},
            ],
        }
    ]


def test_export_removes_duplicated_stime(
    monkeypatch: pytest.MonkeyPatch, exported_json: list[str]
) -> None:
    """A STIME split into several normalized items is exported once."""

    def split_stime_normalization(data: dict, **kwargs: Any) -> dict:
        # '100 ns' normalized into two items, as norm_stime does for intervals.
        result = fake_normalize_json_content(data, **kwargs)
        entities = result["normalized_entities"]["entities"]
        entities.extend(e for e in list(entities) if e["category"] == "STIME")
        return result

    monkeypatch.setattr(
        normalization_module, "normalize_json_content", split_stime_normalization
    )
    run_extraction()

    entities = json.loads(exported_json[-1])[0]["entities"]

    assert entities.count({"100 ns": "STIME"}) == 1


def test_app_starts_without_error() -> None:
    """The app renders its header without any exception."""
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()

    assert not app.exception
    assert "MetaMD" in app.header[0].value


@pytest.mark.parametrize(
    ("default_model", "nb_quota_captions"), [("llm", 1), ("gliner", 0)]
)
def test_quota_is_displayed_in_llm_mode_only(
    monkeypatch: pytest.MonkeyPatch, default_model: str, nb_quota_captions: int
) -> None:
    """The daily quota is shown in LLM mode and hidden in GLiNER mode."""
    monkeypatch.setenv("DEFAULT_MODEL", default_model)

    app = AppTest.from_file(APP_PATH, default_timeout=60).run()

    assert not app.exception
    quota_captions = [c for c in app.caption if "daily quota" in c.value]
    assert len(quota_captions) == nb_quota_captions


def test_extraction_displays_quality_check_and_export() -> None:
    """A successful extraction shows the quality check and enables the export."""
    app = run_extraction()

    assert not app.exception
    assert not app.error
    assert any("Quality Check" in subheader.value for subheader in app.subheader)
    # Only FFM is missing from the fake predictions.
    assert "Missing required metadata categories" in app.warning[0].value
    assert "0 (missing)" in app.warning[0].value
    assert len(app.get("download_button")) == 1


def test_empty_description_shows_a_warning() -> None:
    """Extraction is not run when the description is empty."""
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    app.text_area(key="input_text").input("   ").run()

    app.button[-1].click().run()

    assert not app.exception
    assert app.warning[0].value == (
        "Please provide a description before running extraction."
    )
    assert not app.get("download_button")


def test_extraction_error_is_displayed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An error during extraction is shown to the user instead of crashing."""

    def failing_load_model(*args: Any, **kwargs: Any) -> None:
        msg = "Repository Not Found"
        raise OSError(msg)

    monkeypatch.setattr(gliner_module, "load_model", failing_load_model)
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    app.text_area(key="input_text").input(TEXT).run()

    app.button[-1].click().run()

    assert not app.exception
    assert "Repository Not Found" in app.error[0].value
