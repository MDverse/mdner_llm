"""Module to generate, cache, and load document embeddings via OpenRouter."""

import json
from pathlib import Path

import click
import loguru
import numpy as np
from openai import OpenAI

from mdner_llm.common import load_api_key
from mdner_llm.logger import create_logger


def load_groundtruth_texts(
    groundtruth_dir: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> dict[str, str]:
    """Load 'raw_text' from JSON files in the groundtruth directory.

    Returns
    -------
    dict[str, str]
        Mapping of document filenames to their raw text content.
    """
    # Retrieve all JSON files in the specified directory.
    target_files = [p.name for p in groundtruth_dir.glob("*.json")]
    # Load the 'raw_text' field from each JSON file into a dictionary.
    texts = {}
    for filename in target_files:
        file_path = groundtruth_dir / filename
        with file_path.open(encoding="utf-8") as f:
            data = json.load(f)
            texts[filename] = data["raw_text"]
    logger.info(
        f"Retrieved raw text for {len(texts)} documents from '{groundtruth_dir}'."
    )
    return texts


def fetch_from_api_and_save_embeddings(
    texts_dict: dict[str, str],
    embedding_path: Path,
    embedding_model: str,
    api_key_env: str = "OPENROUTER_API_KEY",
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[list[str], np.ndarray]:
    """Fetch embeddings from OpenRouter API, normalize them, and save to disk.

    Returns
    -------
    tuple[list[str], np.ndarray]
        List of filenames/IDs and their corresponding L2-normalized embeddings.
    """
    # Prepare the list of document identifiers and their corresponding text content.
    filenames = list(texts_dict.keys())
    text_list = list(texts_dict.values())
    logger.info(
        f"Requesting embeddings for {len(text_list)} items using '{embedding_model}'..."
    )
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=load_api_key(api_key_env),
    )
    response = client.embeddings.create(model=embedding_model, input=text_list)
    # Extract raw embeddings and convert to a NumPy array.
    raw_embeds = np.array([item.embedding for item in response.data], dtype=np.float32)
    # L2-normalize vectors for direct cosine similarity via dot product
    norm = np.linalg.norm(raw_embeds, axis=1, keepdims=True)
    norm_embeds = raw_embeds / np.maximum(norm, 1e-12)
    # Save the normalized embeddings to a compressed .npz file.
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        embedding_path,
        filenames=np.array(filenames, dtype=str),
        embeddings=norm_embeds,
    )
    logger.success(f"Saved {len(filenames)} embeddings to '{embedding_path}'.")
    return filenames, norm_embeds


def get_or_create_embeddings(
    texts_dict: dict[str, str],
    embedding_model: str,
    embedding_path: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[list[str], np.ndarray]:
    """Load cached embeddings if available; otherwise compute and persist them.

    Returns
    -------
    tuple[list[str], np.ndarray]
        List of document identifiers and their normalized embedding matrix.
    """
    if embedding_path.exists():
        logger.info(f"Embeddings already exist at '{embedding_path}'.")
        with np.load(embedding_path, allow_pickle=False) as data:
            filenames = data["filenames"].tolist()
            embeddings = data["embeddings"]
        # Validate that cache matches current input keys
        if set(filenames) == set(texts_dict.keys()):
            logger.success(f"Loaded {len(filenames)} embeddings.")
            return filenames, embeddings
        logger.warning("Cache mismatch with current texts. Recomputing...")

    return fetch_from_api_and_save_embeddings(
        texts_dict=texts_dict,
        embedding_path=embedding_path,
        embedding_model=embedding_model,
        logger=logger,
    )


@click.command(name="create-embeddings")
@click.option(
    "--groundtruth-dir-path",
    type=click.Path(exists=True, dir_okay=True, path_type=Path),
    required=True,
    help="Path to the directory containing JSON files with ground truth text.",
)
@click.option(
    "--output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/groundtruth/embeddings.npz"),
    help="Output .npz file path for saved embeddings.",
)
@click.option(
    "--model",
    type=str,
    help="Embedding model name in Openrouter (e.g., 'openai/text-embedding-3-large').",
)
def main(
    groundtruth_dir_path: Path,
    output_path: Path,
    model: str,
) -> None:
    """CLI to generate or load embeddings."""
    # Set up logger.
    logger = create_logger()
    # Load texts from groundtruth JSON files.
    texts_dict = load_groundtruth_texts(groundtruth_dir_path, logger=logger)
    # Generate and save embeddings to the specified output path.
    get_or_create_embeddings(
        texts_dict=texts_dict, output_path=output_path, model=model, logger=logger
    )


if __name__ == "__main__":
    main()
