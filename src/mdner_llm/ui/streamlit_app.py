"""Streamlit app for NER applied to Molecular Dynamics dataset description."""

import re
from itertools import zip_longest
from pathlib import Path

import loguru
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError
from spacy import displacy

from mdner_llm.annotations.colors import COLORS
from mdner_llm.common import load_api_key
from mdner_llm.core.extract_entities_with_llm import (
    annotate_with_llm_and_framework,
    load_prompt,
)
from mdner_llm.logger import create_logger
from mdner_llm.models.app_settings import Settings
from mdner_llm.normalization.normalize_entities import normalize_json_content


def render_spacy_ner(text: str, entities: list[dict]) -> None:
    """Render spaCy-style NER HTML visualization using displacy."""
    spacy_ents = []
    search_from = 0

    for ent in entities:
        ent_text = ent.get("text", "")
        category = ent.get("category", "ENTITY").upper()
        if not ent_text:
            continue
        # Detect the index of SOFTNAME entity
        if category == "SOFTNAME":
            start = text.find(ent_text)
            if start != -1:
                spacy_ents.append(
                    {"start": start, "end": start + len(ent_text), "label": category}
                )
                search_from = start + len(ent_text)
        # Detect the index of SOFTVERS entity after the last SOFTNAME entity
        elif category == "SOFTVERS":
            start = text.find(ent_text, search_from)
            if start != -1:
                spacy_ents.append(
                    {"start": start, "end": start + len(ent_text), "label": category}
                )
        else:
            spacy_ents.extend(
                {"start": match.start(), "end": match.end(), "label": category}
                for match in re.finditer(re.escape(ent_text), text)
            )
    # Remove duplicate spans and sort by start index
    unique_spans = sorted(
        {(entity["start"], entity["end"], entity["label"]) for entity in spacy_ents}
    )
    doc_data = [
        {
            "text": text,
            "ents": [
                {"start": start, "end": end, "label": label}
                for start, end, label in unique_spans
            ],
            "title": None,
        }
    ]
    html = displacy.render(
        doc_data, style="ent", manual=True, options={"colors": COLORS}
    )
    st.write(html, unsafe_allow_html=True)


def run_quality_check(entities: list[dict]) -> None:
    """Verify required metadata categories and flag missing fields."""
    st.subheader("⚠️ Quality Check")
    categories = {ent.get("category", "").upper() for ent in entities}
    checks = [
        ("MOL", "Molecules (`MOL`)"),
        ("FFM", "Force fields or Models (`FFM`)"),
        ("SOFTNAME", "Software names (`SOFTNAME`)"),
        ("SOFTVERS", "Software versions (`SOFTVERS`)"),
        ("STIME", "Simulation time (`STIME`)"),
        ("STEMP", "Simulation temperature (`STEMP`)"),
    ]
    missing_fields = [label for cat_code, label in checks if cat_code not in categories]
    if missing_fields:
        st.warning(
            "**Missing metadata in description:**\n\n"
            + "\n".join([f"- {field}" for field in missing_fields])
        )
    else:
        st.success("✅ All required metadata fields were identified!")


def _show_category_table(cat_df: pd.DataFrame, label: str) -> None:
    """Display a category-specific table with a colored header."""
    # Clean dataframe: drop metadata columns, empty columns, and duplicates
    df = (
        cat_df.drop(columns=["category", "text_normalized"], errors="ignore")
        .dropna(axis=1, how="all")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    # Auto-detect URL columns for interactive Streamlit links
    col_config = {
        col: st.column_config.LinkColumn(col)
        for col in df.columns
        if df[col].astype(str).str.contains(r"^https?://", regex=True).any()
    }
    # Render header badge and dataframe
    color = COLORS.get(label, "#e0e0e0")
    st.markdown(
        f"<div style='background-color:{color};padding:4px 10px;border-radius:6px;"
        f"display:inline-block;font-weight:bold;color:#000;margin:10px 0 4px 0;'>"
        f"{label}</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(df, column_config=col_config, hide_index=True)


def render_parameters_table(norm_entities: list[dict]) -> None:
    """Render the extracted metadata tables."""
    if not norm_entities:
        st.error("No valid entities extracted.")
        return

    df = pd.DataFrame(norm_entities).drop(columns=["is_hallucinated"], errors="ignore")
    st.write(f"**Number of extracted metadata:** {len(df)}")
    # Group categories into pairs
    categories = sorted(df["category"].str.upper().unique())
    for cat1, cat2 in zip_longest(categories[::2], categories[1::2]):
        col1, col2 = st.columns(2)

        with col1:
            _show_category_table(df.query("category.str.upper() == @cat1"), cat1)

        with col2:
            if cat2:
                _show_category_table(df.query("category.str.upper() == @cat2"), cat2)


def _run_extraction_stage(
    llm_model_name: str,
    prompt_path: Path,
    guidelines_path: Path,
    examples_path: Path,
    logger: "loguru.Logger",
) -> None:
    """Execute LLM extraction stage and update session state."""
    st.subheader("📝 Annotation Results")
    with st.spinner("Extracting MD metadata..."):
        try:
            api_key = load_api_key("OPENROUTER_API_KEY")
            prompt = load_prompt(prompt_path, guidelines_path, examples_path, logger)
            extracted_data, _ = annotate_with_llm_and_framework(
                framework="instructor",
                text_to_annotate=st.session_state["pending_text"],
                model=llm_model_name,
                api_key=api_key,
                prompt=prompt,
                provider=None,
                temperature=None,
                logger=logger,
            )
            # Store raw extraction for immediate display
            st.session_state["extracted_entities"] = [
                ent.model_dump() for ent in extracted_data.entities
            ]
            st.session_state["processed_text"] = st.session_state["pending_text"]
            st.session_state["raw_extraction"] = extracted_data.model_dump()
            st.session_state["stage"] = "normalize"
        except (ValueError, KeyError, RuntimeError, ValidationError) as err:
            logger.error(f"Extraction error: {err}")
            st.error(f"An error occurred during extraction: {err}")
            st.session_state["stage"] = None
    st.rerun()


def _run_normalization_stage(
    llm_normalization_model_name: str,
    ffm_db_path: Path,
    softname_db_path: Path,
    logger: "loguru.Logger",
) -> None:
    """Execute entity normalization stage and filter out hallucinations."""
    payload = {
        "text": st.session_state["processed_text"],
        "formatted_response": st.session_state["raw_extraction"],
    }
    normalized_payload = normalize_json_content(
        payload,
        ffm_db_path=ffm_db_path,
        softname_db_path=softname_db_path,
        model_name=llm_normalization_model_name,
        logger=logger,
    )
    if normalized_payload:
        all_norm_ents = normalized_payload.get("normalized_entities", {}).get(
            "entities", []
        )
        st.session_state["normalized_entities"] = [
            ent for ent in all_norm_ents if not ent.get("is_hallucinated", False)
        ]
    else:
        st.error("Error during normalization step.")

    st.session_state["stage"] = "done"
    st.rerun()


def extract_md_metadata(
    llm_model_name: str,
    prompt_path: Path,
    guidelines_path: Path,
    examples_path: Path,
    llm_normalization_model_name: str,
    ffm_db_path: Path,
    softname_db_path: Path,
    logger: "loguru.Logger" = loguru.logger,
):
    """Render Streamlit UI for MD metadata extraction."""
    st.header("⚛︎ MetaMD: AI-Assisted Annotation for MD Simulations")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📥 Input Description")
        dataset_description = st.text_area(
            "Enter Molecular Dynamics dataset description:",
            height=300,
            placeholder=(
                "Example: We ran a 100 ns simulation of a POPC membrane "
                "using GROMACS 2021.4 at 310 K..."
            ),
            key="input_text",
        )
        submit_btn = st.button(
            "Extract Metadata", type="primary", use_container_width=True
        )

        if submit_btn:
            if not dataset_description.strip():
                st.warning("Please provide a description before running extraction.")
            else:
                # Reset state keys before re-running extraction
                for k in (
                    "extracted_entities",
                    "processed_text",
                    "raw_extraction",
                    "normalized_entities",
                ):
                    st.session_state.pop(k, None)
                st.session_state["pending_text"] = dataset_description
                st.session_state["stage"] = "extract"
                st.rerun()

        if "extracted_entities" in st.session_state:
            st.divider()
            run_quality_check(st.session_state["extracted_entities"])

    with col2:
        stage = st.session_state.get("stage")
        # 1. Trigger extraction if queued
        if stage == "extract":
            _run_extraction_stage(
                llm_model_name, prompt_path, guidelines_path, examples_path, logger
            )
        # 2. Display extracted entities immediately in UI tabs
        if "extracted_entities" in st.session_state:
            st.subheader("📝 Annotation Results")
            tab1, tab2 = st.tabs(["🏷️ Visualization", "📋 Extracted Parameters"])
            with tab1:
                render_spacy_ner(
                    st.session_state["processed_text"],
                    st.session_state["extracted_entities"],
                )
            with tab2:
                if "normalized_entities" in st.session_state:
                    render_parameters_table(st.session_state["normalized_entities"])
                else:
                    st.info("Normalizing extracted parameters...")

        # 3. Trigger normalization process
        if stage == "normalize":
            _run_normalization_stage(
                llm_normalization_model_name, ffm_db_path, softname_db_path, logger
            )


def main():
    """Streamlit application entry point."""
    load_dotenv()
    settings = Settings()
    logger = create_logger(settings.log_path)
    if "welcome_logged" not in st.session_state:
        logger.info(f"Welcome to MDNER_LLM v{settings.app_version}!")
        st.session_state["welcome_logged"] = True
    # Configure Streamlit page settings
    st.set_page_config(page_title=settings.app_name, page_icon="⚛️", layout="wide")
    # Load paths from settings
    prompt_path = Path(settings.llm.prompt_path)
    guidelines_path = Path(settings.llm.guidelines_path)
    examples_path = Path(settings.llm.examples_path)
    ffm_db_path = Path(settings.normalization.ffm_db_path)
    softname_db_path = Path(settings.normalization.softname_db_path)
    # Run the main metadata extraction workflow
    extract_md_metadata(
        llm_model_name=settings.llm.model_name,
        prompt_path=prompt_path,
        guidelines_path=guidelines_path,
        examples_path=examples_path,
        llm_normalization_model_name=settings.normalization.llm_model_name,
        ffm_db_path=ffm_db_path,
        softname_db_path=softname_db_path,
        logger=logger,
    )


if __name__ == "__main__":
    main()
