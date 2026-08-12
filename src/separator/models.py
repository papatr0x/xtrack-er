"""Configuration for the demucs separation model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Which demucs checkpoint to load and how to describe it in the UI."""

    name: str
    model_filename: str
    description: str


DEMUCS_MODEL = ModelConfig(
    name="Demucs v4 (htdemucs_6s)",
    model_filename="htdemucs_6s",
    description=(
        "Splits a song into vocals, drums, bass, guitar, piano and other "
        "(a few minutes per song; faster with a GPU or Apple Silicon)"
    ),
)
