"""Streamlit app for Named Entity Recognition on Molecular Dynamics descriptions."""

import json
import time
from pathlib import Path

import httpx
import loguru
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError
from spacy import displacy

from mdner_llm.annotations.categories import CATEGORIES
from mdner_llm.common import load_api_key
from mdner_llm.gliner.extract_entities_with_gliner_all_texts import (
    load_model,
    run_gliner,
)
from mdner_llm.llm.extract_entities_with_llm import (
    annotate_with_llm_and_framework,
    load_prompt,
)
from mdner_llm.logger import create_logger
from mdner_llm.models.entities import ListOfEntities
from mdner_llm.normalization.normalize_entities import (
    clean_text_fields,
    normalize_json_content,
)
from mdner_llm.ui.app_settings import Settings
from mdner_llm.visualization.annotations import convert_annotations_from_model
from mdner_llm.visualization.colors import COLORS

ENTITY_NAME_CATEGORIES = {
    "MOL": "Molecule Name",
    "FFM": "Force Field / Model",
    "SOFTNAME": "Software Name",
    "SOFTVERS": "Software Version",
    "STIME": "Simulation Time",
    "STEMP": "Simulation Temperature",
}


@st.cache_data(ttl=30)
def show_daily_quota(api_key: str, _logger: "loguru.Logger") -> bool:
    """Display the daily quota and return whether requests are allowed.

    Returns
    -------
    bool
        True if requests are allowed, False if the quota is exhausted.
    """
    # Query the OpenRouter API to retrieve the daily quota information.
    try:
        response = httpx.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()["data"]
    except httpx.HTTPError as quota_error:
        _logger.warning(f"Unable to retrieve the OpenRouter quota: {quota_error}")
        st.warning("Unable to retrieve the OpenRouter quota.")
        return True
    # Extract the limit and remaining quota from the API response.
    limit = data.get("limit")
    remaining = data.get("limit_remaining")
    # Check if the daily quota is properly configured.
    if limit is None or remaining is None or data.get("limit_reset") != "daily":
        _logger.warning("OpenRouter daily quota is not configured.")
        st.caption("Remaining daily quota not configured")
        return True
    # Called at most every 30 s thanks to the cache (no log flooding).
    _logger.info(f"OpenRouter remaining daily quota: {remaining:.2f} / {limit:.0f}.")
    # Calculate the progress percentage for the progress bar.
    progress = max(0.0, min(1.0, remaining / limit)) if limit > 0 else 0.0
    st.caption(f"Remaining daily quota: **{remaining:.1f}** / {limit:.0f}")
    st.progress(progress, text=None, width=180)
    # If the remaining quota is exhausted,
    if remaining <= 0:
        _logger.warning("OpenRouter daily quota exhausted.")
        # display a toast notification to inform the user.
        st.toast(
            "No credits left for today. "
            "Please come back tomorrow for more LLM requests."
        )
        # and prevent further requests.
        return False
    # If the quota is available, allow further requests.
    return True


def run_quality_check(
    predictions: ListOfEntities, text: str, normalized_entities: list
) -> None:
    """Validate presence of required categories."""
    st.subheader("⚠️ Quality Check")
    # Locate every occurrence of the extracted entities in the text.
    spans = convert_annotations_from_model(predictions, text, find_all=True)[0]["ents"]
    # Group the occurrences by category, e.g., {'MOL': ['POPC', 'POPC'], ...}.
    texts = {
        code: [text[s["start"] : s["end"]] for s in spans if s["label"] == code]
        for code in ENTITY_NAME_CATEGORIES
    }
    # Map each extracted text to its normalized form(s) from normalize_json_content,
    # e.g., {'gromacs': {'gromacs'}, '295k': {(295, 'k')}, '100 ns': {'100.0 ns'}}.
    # STEMP keeps its raw text in 'text_normalized', so compare its value and unit.
    normalized_forms = {}
    for entity in normalized_entities:
        form = (
            (entity["value"], entity["unit"])
            if entity["category"] == "STEMP" and entity.get("value") is not None
            else entity["text_normalized"]
        )
        normalized_forms.setdefault(clean_text_fields(entity["text"]), set()).add(form)
    # Count the unique normalized forms per category
    # (e.g., 'GROMACS' = 'Gromacs', '295K' = '295 K').
    # Texts without a normalized form fall back to their cleaned text.
    unique_counts = {}
    for code, items in texts.items():
        cleaned_items = [clean_text_fields(item) for item in items]
        unique_counts[code] = len(
            {frozenset(normalized_forms.get(item, {item})) for item in cleaned_items}
        )
    # Format the item counts: total count and unique count in parentheses,
    # e.g., {'MOL': '2 (1)', 'FFM': '0 (missing)', ...}.
    counts = {
        code: f"{len(items)} ({unique_counts[code]})" if items else "0 (missing)"
        for code, items in texts.items()
    }
    # Build a markdown table with one row per category,
    # e.g., '| Molecule Name (`MOL`) | 2 (1) | ✔️ |' or '| ... | 0 (missing) | ❌ |'.
    rows = "\n".join(
        f"| {name} (`{code}`) | {counts[code]} | {'✔️' if texts[code] else '❌'} |"
        for code, name in ENTITY_NAME_CATEGORIES.items()
    )
    table = f"| Category | Items (Unique) | Quality |\n|---|---|---|\n{rows}"
    # Highlight validation status.
    if all(texts.values()):
        st.success(f"**All required metadata categories:**\n\n{table}")
    else:
        st.warning(f"**Missing required metadata categories:**\n\n{table}")


def render_export_button() -> None:
    """Display a button to export the normalized entities as JSON."""
    # Disable the button until the normalization step is done.
    if "normalized_entities" not in st.session_state:
        st.button(
            "Export (JSON)",
            icon=":material/download:",
            disabled=True,
            help="Normalizing extracted parameters...",
            width="stretch",
        )
        return
    # Deduplicate the (text, category) pairs of non-hallucinated entities.
    # A STIME entity may be split into several normalized items (e.g., '0.03 and 3 μs')
    unique_entities = dict.fromkeys(
        (entity["text"], entity["category"])
        for entity in st.session_state["normalized_entities"]
    )
    # Build the export data in the expected format: [{"text": ..., "entities": [...]}].
    export_data = [
        {
            "text": st.session_state["processed_text"],
            "entities": [
                {entity_text: category} for entity_text, category in unique_entities
            ],
        }
    ]
    st.download_button(
        "Export (JSON)",
        data=json.dumps(export_data, indent=2, ensure_ascii=False),
        file_name="metamd_metadata.json",
        mime="application/json",
        on_click=lambda: loguru.logger.info(
            f"Exported {len(export_data[0]['entities'])} entities as JSON."
        ),
        icon=":material/download:",
        type="secondary",
        width="stretch",
    )


def log_extracted_entities(
    predictions: ListOfEntities, inference_time: float, logger: "loguru.Logger"
) -> None:
    """Log the extracted entities, grouped by category, and the inference time."""
    logger.info(
        f"Extracted {len(predictions.entities)} entities in {inference_time:.1f} s."
    )
    # Group the entity texts by category, e.g., {'MOL': ['POPC', 'PSM'], ...}.
    texts_per_category = {
        code: [
            entity.text for entity in predictions.entities if entity.category == code
        ]
        for code in ENTITY_NAME_CATEGORIES
    }
    # Report the missing categories, or confirm that all of them were found.
    missing = [code for code, texts in texts_per_category.items() if not texts]
    if missing:
        logger.warning(f"Missing categories: {', '.join(missing)}.")
    else:
        logger.success("All categories found.")
    # One line per found category, e.g., 'MOL (2): POPC, PSM'.
    for code, texts in texts_per_category.items():
        if texts:
            logger.info(f"{code} ({len(texts)}): {', '.join(texts)}")


def run_llm_extraction(
    model: str,
    framework: str,
    prompt: Path,
    guidelines: Path,
    examples: Path,
    api_key: str,
    logger: "loguru.Logger",
) -> None:
    """Execute entity extraction via large language model inference."""  # noqa: DOC501 (caught below)
    with st.spinner("Extracting MD metadata..."):
        logger.info(f"Starting metadata extraction with the {model} LLM.")
        try:
            # Query LLM to obtain structured annotations.
            llm_response, metadata = annotate_with_llm_and_framework(
                framework=framework,
                text_to_annotate=st.session_state["pending_text"],
                model=model,
                api_key=api_key,
                prompt=load_prompt(prompt, guidelines, examples, logger),
                provider=None,
                temperature=None,
                logger=logger,
            )
            # API and format errors are caught by annotate_with_llm_and_framework,
            # which then returns no entities: check the status to detect them.
            if metadata["status"] != "ok":
                msg = f"LLM request failed ({metadata['status']})."
                raise RuntimeError(msg)
            # Persist output into session state before moving to normalization.
            st.session_state["extracted_entities"] = [
                entity.model_dump() for entity in llm_response.entities
            ]
            st.session_state["processed_text"] = st.session_state["pending_text"]
            st.session_state["structured_predictions"] = ListOfEntities.model_validate(
                llm_response
            )
            # Keep the model name and inference time to display them later.
            st.session_state["extraction_info"] = (
                model,
                metadata["usage"]["inference_time_sec"],
            )
            log_extracted_entities(
                st.session_state["structured_predictions"],
                metadata["usage"]["inference_time_sec"],
                logger,
            )
            st.session_state["stage"] = "normalize"
        except (
            ValueError,
            KeyError,
            RuntimeError,
            ValidationError,
            OSError,
        ) as extraction_error:
            logger.error(f"LLM extraction failed: {extraction_error}")
            st.error(f"An error occurred during extraction: {extraction_error}")
            st.session_state["stage"] = None
            # No rerun on error: it would clear the message before it is seen.
            return
    st.rerun()


def run_gliner_extraction(
    model_name: str,
    model_path: Path,
    device: str,
    logger: "loguru.Logger",
) -> None:
    """Execute entity extraction via Gliner2 model inference."""
    with st.spinner("Extracting MD metadata..."):
        try:
            # Retrieve the Gliner2 model (loaded only on the first call).
            logger.info(
                f"Starting metadata extraction with the {model_name} GLiNER model "
                f"on {device}."
            )
            gliner_model = st.cache_resource(show_spinner=False)(load_model)(
                model_path=model_path, adapter_path=None, device=device
            )
            # Query Gliner2.
            gliner_response, inference_time, _in_tokens, _out_tokens = run_gliner(
                gliner_model,
                st.session_state["pending_text"],
                CATEGORIES,
            )
            # Structure the predictions into a ListOfEntities model.
            structured_gliner_response = ListOfEntities.model_validate(
                {
                    "entities": [
                        {
                            "category": category,
                            "text": ent["text"],
                        }
                        for category, ents in gliner_response.get(
                            "entities", {}
                        ).items()
                        for ent in ents
                    ]
                }
            )
            # Persist output into session state before moving to normalization.
            st.session_state["extracted_entities"] = [
                entity.model_dump() for entity in structured_gliner_response.entities
            ]
            st.session_state["processed_text"] = st.session_state["pending_text"]
            st.session_state["structured_predictions"] = structured_gliner_response
            # Keep the model name and inference time to display them later.
            st.session_state["extraction_info"] = (model_name, inference_time)
            log_extracted_entities(structured_gliner_response, inference_time, logger)
            st.session_state["stage"] = "normalize"
        except (
            ValueError,
            KeyError,
            RuntimeError,
            ValidationError,
            OSError,
        ) as extraction_error:
            logger.error(
                f"GLiNER extraction failed ({type(extraction_error).__name__}): "
                f"{extraction_error}"
            )
            st.error(f"An error occurred during Gliner2 extraction: {extraction_error}")
            st.session_state["stage"] = None
            # No rerun on error: it would clear the message before it is seen.
            return
    st.rerun()


def run_extraction(
    settings: Settings,
    model: str,
    prompt: Path,
    guidelines: Path,
    examples: Path,
    api_key: str | None,
    logger: "loguru.Logger",
) -> None:
    """Execute entity extraction with the configured model (LLM or GLiNER)."""
    # GLiNER mode: run the Gliner2 model (no daily quota involved).
    if settings.default_model == "gliner":
        run_gliner_extraction(
            settings.gliner.model_name,
            settings.gliner.model_path,
            settings.gliner.device,
            logger,
        )
    # LLM mode: run the LLM only if the daily quota is not exhausted.
    elif st.session_state["is_llm_requests_allowed"]:
        run_llm_extraction(
            model,
            settings.llm.framework,
            prompt,
            guidelines,
            examples,
            api_key,
            logger,
        )
    else:
        logger.warning("LLM extraction refused: daily quota exhausted.")
        st.error(
            "No credits left for today. "
            "Please come back tomorrow for more LLM requests."
        )
        st.session_state["stage"] = None


def run_normalization(
    model: str,
    ffm_path: Path,
    soft_path: Path,
    api_key: str,
    logger: "loguru.Logger",
) -> None:
    """Execute normalization and remove hallucinated entities."""
    logger.info("Starting normalization of the extracted entities.")
    start_time = time.perf_counter()
    # Normalize extracted entities against reference databases.
    result = normalize_json_content(
        {
            "text": st.session_state["processed_text"],
            "formatted_response": st.session_state["structured_predictions"],
        },
        ffm_db_path=ffm_path,
        softname_db_path=soft_path,
        model_name=model,
        logger=logger,
        api_key=api_key,
    )
    # Filter out entities flagged as hallucinations by the LLM.
    if result:
        entities = result.get("normalized_entities", {}).get("entities", [])
        hallucinated = [entity for entity in entities if entity.get("is_hallucinated")]
        st.session_state["normalized_entities"] = [
            entity for entity in entities if not entity.get("is_hallucinated")
        ]
        logger.info(f"Normalization done in {time.perf_counter() - start_time:.1f} s.")
        # Report the hallucinated entities removed from the result.
        if hallucinated:
            removed = ", ".join(
                f"{entity['text']} ({entity['category']})" for entity in hallucinated
            )
            logger.warning(
                f"{len(hallucinated)} hallucinated entities removed: {removed}"
            )
        else:
            logger.info("No hallucinated entities.")
    else:
        logger.error("Normalization failed: normalize_json_content returned None.")
        # Fall back on the raw extracted entities so that the quality check
        # and the export button are still available.
        st.session_state["normalized_entities"] = [
            {**entity, "text_normalized": clean_text_fields(entity["text"])}
            for entity in st.session_state["extracted_entities"]
        ]
    st.session_state["stage"] = "done"
    st.rerun()


def run_app(
    settings: Settings,
    model: str,
    prompt: Path,
    guidelines: Path,
    examples: Path,
    ffm_path: Path,
    soft_path: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Build and render the Streamlit metadata extraction interface."""
    # Display the header (and the daily quota in LLM mode) in a two-column layout.
    title_column, quota_column = st.columns([10, 1])
    with title_column:
        st.header(f"⚛︎ {settings.app_name}: {settings.app_short_description}")
    # Retrieve the OpenRouter API key from environment variables.
    # It is needed in both modes: the simulation times (STIME) are
    # normalized with an LLM, so they also count in the daily quota.
    api_key = load_api_key("OPENROUTER_STREAMLIT_API_KEY")
    with quota_column:
        st.session_state["is_llm_requests_allowed"] = show_daily_quota(api_key, logger)
    # Render two-column layout for input and output sections.
    left_column, right_column = st.columns([1, 1], gap="large")
    # Input section: Text area for user to provide MD dataset description.
    with left_column:
        st.subheader("📥 Input Description")
        description = st.text_area(
            "Enter Molecular Dynamics dataset description:",
            height=300,
            placeholder=(
                "Example: We ran a 100 ns simulation of a POPC membrane "
                "using GROMACS 2021.4 at 310 K..."
            ),
            key="input_text",
        )
        # Examples for the user to choose from.
        example_columns = st.columns(3)
        for column, (key, example) in zip(
            example_columns, settings.examples.items(), strict=False
        ):
            with column:
                st.button(
                    key.replace("_", " ").title(),
                    key=f"button_{key}",
                    help=f"Source: {example['source']}",
                    on_click=lambda text=example["text"], name=key: (
                        logger.info(f"Example '{name}' loaded in the input."),
                        st.session_state.update({"input_text": text}),
                    ),
                    width="stretch",
                )
        # Button to trigger extraction.
        extract_clicked = st.button("Extract Metadata", type="primary", width="stretch")
        # Disclaimer displayed right below the extraction button.
        st.caption(
            ":material/info: Metadata extraction is AI-assisted. "
            "Please check for inconsistencies."
        )
        if extract_clicked:
            if not description.strip():
                logger.warning("Extraction requested with an empty description.")
                st.warning("Please provide a description before running extraction.")
            else:
                logger.info(
                    f"Text to annotate ({len(description)} characters):\n"
                    f"{description.strip()}"
                )
                # Clear previous session state.
                for key_name in (
                    "extracted_entities",
                    "processed_text",
                    "structured_predictions",
                    "normalized_entities",
                    "extraction_info",
                ):
                    st.session_state.pop(key_name, None)
                # Store the input description and set the stage for extraction.
                st.session_state["pending_text"] = description
                st.session_state["stage"] = "extract"
                # Force rerun to trigger extraction logic in the right column.
                st.rerun()
        # Display quality check results once entities have been normalized.
        if "normalized_entities" in st.session_state:
            st.divider()
            # Run quality check on extracted entities and display results.
            run_quality_check(
                st.session_state["structured_predictions"],
                st.session_state["processed_text"],
                st.session_state["normalized_entities"],
            )
            # Display which model extracted the metadata and how long it took.
            model_name, inference_time = st.session_state["extraction_info"]
            st.caption(
                f"Metadata extracted with **{model_name}** "
                f"in **{inference_time:.1f} seconds**."
            )

    # Output section: Display extraction results and normalization options.
    with right_column:
        current_stage = st.session_state.get("stage")
        # Step 1: Run LLM/gliner extraction.
        if current_stage == "extract":
            st.subheader("📝 Annotation Results")
            run_extraction(
                settings, model, prompt, guidelines, examples, api_key, logger
            )
        # Step 2: Render results tabs.
        if "extracted_entities" in st.session_state:
            # Title on the left, export button on the top right.
            title_column, export_column = st.columns(
                [3, 1], vertical_alignment="center"
            )
            with title_column:
                st.subheader("📝 Annotation Results")
            with export_column:
                # Button to export the extracted metadata as JSON.
                render_export_button()
            # Create the visualization tab.
            (viz_tab,) = st.tabs(["🏷️ Visualization"])
            # Highlight entities in the text.
            with viz_tab:
                # Convert LLM output into displaCy-compatible format.
                converted_data = convert_annotations_from_model(
                    st.session_state["structured_predictions"],
                    st.session_state["processed_text"],
                    find_all=True,
                )
                # Restore the original text to keep its line breaks
                # (same length, so entity positions stay valid).
                converted_data[0]["text"] = st.session_state["processed_text"]
                # Render text with highlighted entities.
                html_content = displacy.render(
                    converted_data, style="ent", manual=True, options={"colors": COLORS}
                )
                st.write(html_content, unsafe_allow_html=True)

        # Step 3: Run normalization on extracted entities.
        if current_stage == "normalize":
            run_normalization(model, ffm_path, soft_path, api_key, logger)


def main() -> None:
    """Execute main entry point for the Streamlit dashboard."""
    # Load environment variables.
    load_dotenv()
    # Load application settings (cached across reruns).
    settings = st.cache_resource(show_spinner=False)(Settings)()
    # Initialize logger (cached across reruns).
    app_logger = st.cache_resource(show_spinner=False)(create_logger)(settings.log_path)
    # Launch main UI.
    st.set_page_config(
        page_title=settings.app_name,
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    # Log the welcome message once per user session.
    if "session_started" not in st.session_state:
        st.session_state["session_started"] = True
        app_logger.info(f"Welcome to {settings.app_name} v{settings.app_version}")
    # Create a sidebar for application settings and information.
    with st.sidebar:
        st.image("img/logo_app.png")
        st.title(f"**{settings.app_name}** **v{settings.app_version}**")
        st.markdown(f"{settings.app_long_description}")
        st.divider()
        st.markdown("**Contact:** [MDverse community]()")  # TO-ADD
        st.markdown("**Code source:** https://github.com/MDverse/mdner_llm")
    run_app(
        settings=settings,
        model=settings.llm.model_name,
        prompt=settings.llm.prompt_path,
        guidelines=settings.llm.guidelines_path,
        examples=settings.llm.examples_path,
        ffm_path=settings.normalization.ffm_db_path,
        soft_path=settings.normalization.softname_db_path,
        logger=app_logger,
    )


if __name__ == "__main__":
    main()
