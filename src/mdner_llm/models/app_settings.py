"""Pydantic data models used to validate UI Streamlit application settings."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, FilePath, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class NormalizationConfig(BaseModel):
    """Configuration model for normalization database paths."""

    llm_model_name: str = Field(
        ..., description="LLM model name to use for normalization tasks."
    )
    ffm_db_path: FilePath = Field(
        ..., description="Path to the FFM normalization database file."
    )
    softname_db_path: FilePath = Field(
        ..., description="Path to the SoftName normalization database file."
    )


class LLMConfig(BaseModel):
    """Configuration model for Large Language Model (LLM) parameters."""

    model_name: str = Field(
        ..., description="Default LLM model name to use if none is provided."
    )
    framework: Literal["instructor", "noframework", "pydanticai"] = Field(
        ..., description="Framework to use for validating the LLM response format."
    )
    prompt_path: FilePath = Field(
        ..., description="Path to the prompt template file for LLM input."
    )
    guidelines_path: FilePath = Field(
        ..., description="Path to the guidelines file for LLM input."
    )
    examples_path: FilePath = Field(
        ..., description="Path to the examples file for LLM input."
    )


class Settings(BaseSettings, cli_parse_args=True):
    """Global MDNER-LLM streamlit application settings."""

    # Basic application info
    app_name: str = Field(
        ..., description="Application name displayed in the UI or logs."
    )
    app_description: str | None = Field(
        None, description="Short app description displayed in the logs."
    )
    app_version: str = Field(..., description="Current version of the application.")
    # Styling and assets
    css_path: FilePath | None = Field(
        None, description="Path to the CSS file for Streamlit styling."
    )
    openrouter_api_key: SecretStr = Field(
        ...,
        validation_alias=AliasChoices("openrouter_api_key", "OPENROUTER_API_KEY"),
        description="Secret API key for authenticating on openrouter provider.",
    )
    # Nested configuration for LLM-related parameters
    llm: LLMConfig = Field(
        ..., description="Configuration for Large Language Model parameters."
    )
    # Nested configuration for normalization database paths
    normalization: NormalizationConfig = Field(
        ..., description="Configuration for normalization database paths."
    )

    # Compute the log path
    @property
    def log_path(self) -> Path:
        """Return the full path for today's log file."""
        log_file = (
            Path("logs")
            / f"{datetime.now(UTC).strftime('%Y%m%d')}"
            / "mdner_llm_app.log"
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)
        return log_file

    # Load settings from this TOML file
    model_config = SettingsConfigDict(
        toml_file="src/mdner_llm/ui/config_app.toml", extra="ignore"
    )

    # Reorder the priority of different settings sources
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Customize the order and type of settings sources.

        Parameters
        ----------
        settings_cls : type[BaseSettings]
            The settings class being instantiated. Used by sources that
            need access to the settings schema.
        init_settings : PydanticBaseSettingsSource
            Source providing values passed directly to the Settings
            constructor.
        env_settings : PydanticBaseSettingsSource
            Source providing values from environment variables.

        Returns
        -------
        tuple[PydanticBaseSettingsSource, ...]
            Ordered tuple of settings sources. Earlier sources have higher
            priority during value resolution.
        """
        return (
            # Load settings from TOML files first
            TomlConfigSettingsSource(settings_cls),
            # Override with explicit constructor arguments
            init_settings,
            # Finally allow environment variables to override everything
            env_settings,
            dotenv_settings,
        )
