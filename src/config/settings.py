"""Configuration classes for data paths, splitting, models, and experiments."""

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings


class DataPaths(BaseSettings):
    """File paths for the raw input datasets."""

    data_directory: Path = Path("data/original")
    transactions_file: str = "transactions.csv"
    close_prices_file: str = "close_prices.csv"
    customer_information_file: str = "customer_information.csv"
    asset_information_file: str = "asset_information.csv"
    markets_file: str = "markets.csv"


class RandomForestConfig(BaseSettings):
    """Hyperparameters for the Random Forest profitability prediction baseline."""

    number_of_estimators: int = 20
    max_depth: int | None = None
    random_state: int | None = None
    prediction_horizon_months: int = 6


class LightGCNConfig(BaseSettings):
    """Hyperparameters for the LightGCN collaborative filtering model."""

    embedding_dimension: int = 64
    number_of_layers: int = 3
    learning_rate: float = 0.01
    weight_decay: float = 1e-5
    keep_probability: float = 0.6
    number_of_epochs: int = 50
    batch_size: int = 1024


class SASRecConfig(BaseSettings):
    """Hyperparameters for the SASRec sequential recommender."""

    embedding_dimension: int = 64
    max_sequence_length: int = 50
    number_of_attention_heads: int = 2
    number_of_blocks: int = 2
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3
    number_of_epochs: int = 200
    batch_size: int = 128

    @model_validator(mode="after")
    def validate_embedding_divisible_by_heads(self) -> "SASRecConfig":
        """Reject configs where the embedding dimension does not evenly split across heads."""
        if self.embedding_dimension % self.number_of_attention_heads != 0:
            raise ValueError(
                f"embedding_dimension ({self.embedding_dimension}) must be"
                f" divisible by number_of_attention_heads"
                f" ({self.number_of_attention_heads})"
            )
        return self


class TiSASRecConfig(BaseSettings):
    """Hyperparameters for the TiSASRec time-aware sequential recommender."""

    embedding_dimension: int = 64
    max_sequence_length: int = 50
    number_of_attention_heads: int = 2
    number_of_blocks: int = 2
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3
    number_of_epochs: int = 200
    batch_size: int = 128
    time_bucket_count: int = 256

    @model_validator(mode="after")
    def validate_embedding_divisible_by_heads(self) -> "TiSASRecConfig":
        """Reject configs where the embedding dimension does not evenly split across heads."""
        if self.embedding_dimension % self.number_of_attention_heads != 0:
            raise ValueError(
                f"embedding_dimension ({self.embedding_dimension}) must be"
                f" divisible by number_of_attention_heads"
                f" ({self.number_of_attention_heads})"
            )
        return self


class HybridDualHeadConfig(BaseSettings):
    """Hyperparameters for the hybrid dual-head recommender."""

    embedding_dimension: int = 64
    max_sequence_length: int = 50
    number_of_attention_heads: int = 2
    number_of_blocks: int = 2
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3
    number_of_epochs: int = 200
    batch_size: int = 128
    time_bucket_count: int = 256
    profitability_hidden_dimension: int = 64
    loss_lambda: float = 0.5
    inference_alpha: float = 0.5

    @model_validator(mode="after")
    def validate_embedding_divisible_by_heads(self) -> "HybridDualHeadConfig":
        """Reject configs where the embedding dimension does not evenly split across heads."""
        if self.embedding_dimension % self.number_of_attention_heads != 0:
            raise ValueError(
                f"embedding_dimension ({self.embedding_dimension}) must be"
                f" divisible by number_of_attention_heads"
                f" ({self.number_of_attention_heads})"
            )
        return self


class ExperimentConfig(BaseSettings):
    """Top-level experiment settings: device, seed, and evaluation k."""

    top_k: int = 10
    device: str = "cuda"
    seed: int = 42


ModelConfig = (
    RandomForestConfig
    | LightGCNConfig
    | SASRecConfig
    | TiSASRecConfig
    | HybridDualHeadConfig
)
