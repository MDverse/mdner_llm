"""Streamlit app for Named Entity Recognition on Molecular Dynamics descriptions."""

from pathlib import Path

import loguru
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError
from spacy import displacy

from mdner_llm.annotations.colors import COLORS
from mdner_llm.annotations.visualize_annotations import convert_annotations_from_llm
from mdner_llm.common import load_api_key
from mdner_llm.core.extract_entities_with_llm import (
    annotate_with_llm_and_framework,
    load_prompt,
)
from mdner_llm.logger import create_logger
from mdner_llm.models.app_settings import Settings
from mdner_llm.models.entities import ListOfEntities
from mdner_llm.normalization.normalize_entities import normalize_json_content

METADATA_FIELDS = {
    "MOL": "Molecule Name",
    "FFM": "Force Field / Model",
    "SOFTNAME": "Software Name",
    "SOFTVERS": "Software Version",
    "STIME": "Simulation Time",
    "STEMP": "Simulation Temperature",
}


def run_quality_check(entities: list) -> None:
    """Validate presence of required categories."""
    st.subheader("⚠️ Quality Check")
    categories = {entity.get("category", "").upper() for entity in entities}
    # Build status bullet lines, e.g., '- ✔️ **Molecule Name (`MOL`)** : Present'.
    bullets = "\n".join(
        f"- **{name} (`{code}`)** {'✔️' if code in categories else '❌'}"
        for code, name in METADATA_FIELDS.items()
    )
    # Highlight validation status.
    if all(code in categories for code in METADATA_FIELDS):
        st.success(f"**All required metadata categories identified!**\n\n{bullets}")
    else:
        st.warning(f"**Missing required metadata categories:**\n\n{bullets}")


def show_category_table(dataframe: pd.DataFrame, label: str) -> None:
    """Render a category table with a colored label badge."""
    clean_df = (
        # Remove duplicate rows based on normalized text.
        dataframe.drop_duplicates(subset=["text_normalized"], keep="first")
        # Clean unwanted technical columns and empty values.
        .reset_index(drop=True)
        .drop(columns=["category", "text_normalized"], errors="ignore")
        .dropna(axis=1, how="all")
    )
    color = COLORS.get(label, "#e0e0e0")
    title = METADATA_FIELDS.get(label, label)
    # Render inline colored badge and header title.
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;margin:10px 0 6px 0;'>"
        f"<span style='background-color:{color};padding:3px 8px;border-radius:6px;"
        f"font-weight:bold;color:#000;font-size:0.85rem;'>{label}</span>"
        f"<span style='font-weight:600;font-size:1rem;'>{title}</span></div>",
        unsafe_allow_html=True,
    )
    st.dataframe(clean_df, hide_index=True)


def render_parameters_table(entities: list) -> None:
    """Display extracted parameters grouped in a two-column layout."""
    if not entities:
        st.error("No valid entities extracted.")
        return
    dataframe = pd.DataFrame(entities).drop(
        columns=["is_hallucinated"], errors="ignore"
    )
    st.write(f"**Number of extracted metadata:** {len(dataframe)}")
    cols = st.columns(2)
    # Group entities by category and render each group in alternating columns.
    for i, (category, group_df) in enumerate(dataframe.groupby(dataframe["category"])):
        with cols[i % 2]:
            show_category_table(group_df, category)


def run_llm_extraction(
    model: str,
    prompt: Path,
    guidelines: Path,
    examples: Path,
    logger: "loguru.Logger",
) -> None:
    """Execute entity extraction via large language model inference."""
    st.subheader("📝 Annotation Results")
    with st.spinner("Extracting MD metadata..."):
        try:
            # Query LLM to obtain structured annotations.
            output, _metadata = annotate_with_llm_and_framework(
                framework="instructor",
                text_to_annotate=st.session_state["pending_text"],
                model=model,
                api_key=load_api_key("OPENROUTER_API_KEY"),
                prompt=load_prompt(prompt, guidelines, examples, logger),
                provider=None,
                temperature=None,
                logger=logger,
            )
            # Persist output into session state before moving to normalization.
            st.session_state["extracted_entities"] = [
                entity.model_dump() for entity in output.entities
            ]
            st.session_state["processed_text"] = st.session_state["pending_text"]
            st.session_state["raw_extraction"] = ListOfEntities.model_validate(output)
            st.session_state["stage"] = "normalize"
        except (
            ValueError,
            KeyError,
            RuntimeError,
            ValidationError,
        ) as extraction_error:
            logger.error(f"Extraction error: {extraction_error}")
            st.error(f"An error occurred during extraction: {extraction_error}")
            st.session_state["stage"] = None
    st.rerun()


def run_normalization(
    model: str,
    ffm_path: Path,
    soft_path: Path,
    logger: "loguru.Logger",
) -> None:
    """Execute normalization and remove hallucinated entities."""
    payload = {
        "text": st.session_state["processed_text"],
        "formatted_response": st.session_state["raw_extraction"],
    }
    # Normalize extracted entities against reference databases.
    result = normalize_json_content(
        payload,
        ffm_db_path=ffm_path,
        softname_db_path=soft_path,
        model_name=model,
        logger=logger,
    )
    # Filter out entities flagged as hallucinations by the LLM.
    if result:
        st.session_state["normalized_entities"] = [
            entity
            for entity in result.get("normalized_entities", {}).get("entities", [])
            if not entity.get("is_hallucinated", False)
        ]
    else:
        st.error("Error during normalization step.")
    st.session_state["stage"] = "done"
    st.rerun()


def extract_md_metadata(
    model: str,
    prompt: Path,
    guidelines: Path,
    examples: Path,
    norm_model: str,
    ffm_path: Path,
    soft_path: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Build and render the Streamlit metadata extraction interface."""
    st.header(
        "⚛︎ MetaMD: AI-Assisted Annotation for Molecular Dynamics Datasets Descriptions"
    )
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
        # Button to trigger extraction.
        if st.button("Extract Metadata", type="primary", width="stretch"):
            if not description.strip():
                st.warning("Please provide a description before running extraction.")
            else:
                # Clear previous session state.
                for key_name in (
                    "extracted_entities",
                    "processed_text",
                    "raw_extraction",
                    "normalized_entities",
                ):
                    st.session_state.pop(key_name, None)
                # Store the input description and set the stage for extraction.
                st.session_state["pending_text"] = description
                st.session_state["stage"] = "extract"
                # Force rerun to trigger extraction logic in the right column.
                st.rerun()
        # Display quality check results if entities have been extracted.
        if "extracted_entities" in st.session_state:
            st.divider()
            # Run quality check on extracted entities and display results.
            run_quality_check(st.session_state["extracted_entities"])

    # Output section: Display extraction results and normalization options.
    with right_column:
        current_stage = st.session_state.get("stage")
        # Step 1: Run LLM extraction.
        if current_stage == "extract":
            run_llm_extraction(model, prompt, guidelines, examples, logger)
        # Step 2: Render results tabs.
        if "extracted_entities" in st.session_state:
            st.subheader("📝 Annotation Results")
            # Create two tabs.
            viz_tab, params_tab = st.tabs(
                ["🏷️ Visualization", "📋 Extracted Parameters"]
            )
            # First, with highlighted entities in the text.
            with viz_tab:
                # Convert LLM output into displaCy-compatible format.
                converted_data = convert_annotations_from_llm(
                    st.session_state["raw_extraction"],
                    st.session_state["processed_text"],
                )
                # Render text with highlighted entities.
                html_content = displacy.render(
                    converted_data, style="ent", manual=True, options={"colors": COLORS}
                )
                st.write(html_content, unsafe_allow_html=True)
            # Second, with a structured table of extracted parameters.
            with params_tab:
                if "normalized_entities" in st.session_state:
                    render_parameters_table(st.session_state["normalized_entities"])
                else:
                    st.info("Normalizing extracted parameters...")

        # Step 3: Run normalization on extracted entities.
        if current_stage == "normalize":
            run_normalization(norm_model, ffm_path, soft_path, logger)


def main() -> None:
    """Execute main entry point for the Streamlit dashboard."""
    # Load environment variables.
    load_dotenv()
    # Load application settings.
    settings = Settings()
    # Initialize logger.
    app_logger = create_logger(settings.log_path)
    # Launch main UI.
    st.set_page_config(page_title=settings.app_name, page_icon="⚛️", layout="wide")
    extract_md_metadata(
        model=settings.llm.model_name,
        prompt=Path(settings.llm.prompt_path),
        guidelines=Path(settings.llm.guidelines_path),
        examples=Path(settings.llm.examples_path),
        norm_model=settings.llm.model_name,
        ffm_path=Path(settings.normalization.ffm_db_path),
        soft_path=Path(settings.normalization.softname_db_path),
        logger=app_logger,
    )


if __name__ == "__main__":
    main()
