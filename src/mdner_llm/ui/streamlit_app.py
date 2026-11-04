"""Streamlit app for Named Entity Recognition on Molecular Dynamics descriptions."""

from pathlib import Path

import httpx
import loguru
import pandas as pd
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
from mdner_llm.normalization.normalize_entities import normalize_json_content
from mdner_llm.ui.app_settings import Settings
from mdner_llm.visualization.annotations import convert_annotations_from_model
from mdner_llm.visualization.colors import COLORS

ENTITY_CATEGORIES = {
    "MOL": "Molecule Name",
    "FFM": "Force Field / Model",
    "SOFTNAME": "Software Name",
    "SOFTVERS": "Software Version",
    "STIME": "Simulation Time",
    "STEMP": "Simulation Temperature",
}


@st.cache_data(ttl=30)
def show_daily_quota(api_key: str) -> bool:
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
    except httpx.HTTPError:
        st.warning("Unable to retrieve the OpenRouter quota.")
        return True
    # Extract the limit and remaining quota from the API response.
    limit = data.get("limit")
    remaining = data.get("limit_remaining")
    # Check if the daily quota is properly configured.
    if limit is None or remaining is None or data.get("limit_reset") != "daily":
        st.caption("Daily quota not configured")
        return True
    # Calculate the progress percentage for the progress bar.
    progress = max(0.0, min(1.0, remaining / limit)) if limit > 0 else 0.0
    st.caption(f"Daily quota: **{remaining:.1f}** / {limit:.0f}")
    st.progress(progress, text=None, width=150)
    # If the remaining quota is exhausted,
    if remaining <= 0:
        # display a toast notification to inform the user.
        st.toast(
            "No credits left for today. "
            "Please come back tomorrow for more LLM requests."
        )
        # and prevent further requests.
        return False
    # If the quota is available, allow further requests.
    return True


def run_quality_check(entities: list) -> None:
    """Validate presence of required categories."""
    st.subheader("⚠️ Quality Check")
    # Extract unique categories from the extracted entities.
    categories = {entity.get("category", "").upper() for entity in entities}
    # Build status bullet lines, e.g., '- ✔️ **Molecule Name (`MOL`)** : Present'.
    bullets = "\n".join(
        f"- **{name} (`{code}`)** {'✔️' if code in categories else '❌'}"
        for code, name in ENTITY_CATEGORIES.items()
    )
    # Highlight validation status.
    if all(code in categories for code in ENTITY_CATEGORIES):
        st.success(f"**All required metadata categories identified!**\n\n{bullets}")
    else:
        st.warning(f"**Missing required metadata categories:**\n\n{bullets}")


def render_parameters_table(valid_entities: list) -> None:
    """Display extracted parameters grouped in a two-column layout."""
    if not valid_entities:
        st.error("No valid entities extracted.")
        return
    # Convert the list of entities into a DataFrame.
    dataframe = pd.DataFrame(valid_entities).drop(
        columns=["is_hallucinated"], errors="ignore"
    )
    st.write(f"**Number of extracted metadata:** {len(dataframe)}")
    cols = st.columns(2)
    # Render categories in the order defined by ENTITY_CATEGORIES.
    for i, category in enumerate(ENTITY_CATEGORIES):
        df_by_category = dataframe[dataframe["category"] == category]
        if df_by_category.empty:
            continue
        with cols[i % 2]:
            clean_df = (
                # Remove duplicate rows based on normalized text.
                df_by_category.drop_duplicates(subset=["text_normalized"], keep="first")
                # Clean unwanted technical columns and empty values.
                .reset_index(drop=True)
                .drop(columns=["category", "text_normalized"], errors="ignore")
                .dropna(axis=1, how="all")
            )
            color = COLORS.get(category, "#e0e0e0")
            title = ENTITY_CATEGORIES.get(category, category)
            # Render inline colored badge and header title.
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin:10px 0 6px 0;'>"
                f"<span style='background-color:{color};padding:3px 8px;border-radius:6px;"
                f"font-weight:bold;color:#000;font-size:0.85rem;'>{category}</span>"
                f"<span style='font-weight:600;font-size:1rem;'>{title}</span></div>",
                unsafe_allow_html=True,
            )
            st.dataframe(clean_df, hide_index=True)


def run_llm_extraction(
    model: str,
    prompt: Path,
    guidelines: Path,
    examples: Path,
    api_key: str,
    logger: "loguru.Logger",
) -> None:
    """Execute entity extraction via large language model inference."""
    with st.spinner("Extracting MD metadata..."):
        try:
            # Query LLM to obtain structured annotations.
            llm_response, _metadata = annotate_with_llm_and_framework(
                framework="instructor",
                text_to_annotate=st.session_state["pending_text"],
                model=model,
                api_key=api_key,
                prompt=load_prompt(prompt, guidelines, examples, logger),
                provider=None,
                temperature=None,
                logger=logger,
            )
            # Persist output into session state before moving to normalization.
            st.session_state["extracted_entities"] = [
                entity.model_dump() for entity in llm_response.entities
            ]
            st.session_state["processed_text"] = st.session_state["pending_text"]
            st.session_state["structured_predictions"] = ListOfEntities.model_validate(
                llm_response
            )
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


def run_gliner_extraction(
    model_name: str,
    model_path: Path,
    device: str,
    logger: "loguru.Logger",
) -> None:
    """Execute entity extraction via Gliner2 model inference."""
    with st.spinner("Extracting MD metadata..."):
        try:
            # Load Gliner2 model configuration.
            logger.info(f"Loading {model_name} model.")
            gliner_model = load_model(
                model_path=model_path, adapter_path=None, device=device, logger=logger
            )
            # Query Gliner2.
            gliner_response, _inference_time, _in_tokens, _out_tokens = run_gliner(
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
            st.session_state["stage"] = "normalize"
        except (
            ValueError,
            KeyError,
            RuntimeError,
            ValidationError,
        ) as extraction_error:
            logger.error(f"Gliner2 extraction error: {extraction_error}")
            st.error(f"An error occurred during Gliner2 extraction: {extraction_error}")
            st.session_state["stage"] = None
    st.rerun()


def run_normalization(
    model: str,
    ffm_path: Path,
    soft_path: Path,
    logger: "loguru.Logger",
) -> None:
    """Execute normalization and remove hallucinated entities."""
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
    # Retrieve the OpenRouter API key from environment variables.
    api_key = load_api_key("OPENROUTER_STREAMLIT_API_KEY")
    # Display the daily quota in the top-right corner of the app.
    title_column, quota_column = st.columns([10, 1])
    with title_column:
        # Display the main header for the application.
        st.header(f"⚛︎ {settings.app_name}: {settings.app_short_description}")
    with quota_column:
        st.session_state["is_llm_requests_allowed"] = show_daily_quota(api_key)

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
                    "structured_predictions",
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
        # Step 1: Run LLM/gliner extraction.
        if current_stage == "extract":
            st.subheader("📝 Annotation Results")
            if (
                settings.default_model == "llm"
                and st.session_state["is_llm_requests_allowed"]
            ):
                run_llm_extraction(model, prompt, guidelines, examples, api_key, logger)
            else:
                run_gliner_extraction(
                    settings.gliner.model_name,
                    settings.gliner.model_path,
                    settings.gliner.device,
                    logger,
                )
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
                converted_data = convert_annotations_from_model(
                    st.session_state["structured_predictions"],
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
            run_normalization(model, ffm_path, soft_path, logger)


def main() -> None:
    """Execute main entry point for the Streamlit dashboard."""
    # Load environment variables.
    load_dotenv()
    # Load application settings.
    settings = Settings()
    # Initialize logger.
    app_logger = create_logger(settings.log_path)
    # Launch main UI.
    st.set_page_config(
        page_title=settings.app_name,
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    # Create a sidebar for application settings and information.
    with st.sidebar:
        st.image("img/logo_app.jpeg")
        st.title(f"**{settings.app_name}** **V{settings.app_version}**")
        st.markdown(f"{settings.app_long_description}")
        st.divider()
        st.markdown("**Contact:** [MDverse community]()")  # TO-ADD
        st.markdown("**Code source:** https://github.com/MDverse/mdner_llm")
    run_app(
        settings=settings,
        model=settings.llm.model_name,
        prompt=Path(settings.llm.prompt_path),
        guidelines=Path(settings.llm.guidelines_path),
        examples=Path(settings.llm.examples_path),
        ffm_path=Path(settings.normalization.ffm_db_path),
        soft_path=Path(settings.normalization.softname_db_path),
        logger=app_logger,
    )


if __name__ == "__main__":
    main()
