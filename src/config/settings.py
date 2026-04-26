"""Configuration classes for data paths, models, and experiments."""

from pathlib import Path

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
    random_state: int | None = 42
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


class ProfileCoherentLightGCNConfig(BaseSettings):
    """Hyperparameters for the Profile-Coherent LightGCN extension."""

    embedding_dimension: int = 64
    number_of_layers: int = 3
    learning_rate: float = 0.01
    weight_decay: float = 1e-5
    keep_probability: float = 0.6
    number_of_epochs: int = 50
    batch_size: int = 1024

    profile_embedding_dimension: int = 16
    profile_embedding_enabled: bool = True

    profile_coherence_enabled: bool = True
    profile_coherence_lambda: float = 0.5
    profile_coherence_squared: bool = False


class ExperimentConfig(BaseSettings):
    """Top-level experiment settings: device, seed, and evaluation k."""

    top_k: int = 10
    device: str = "cuda"
    seed: int = 42


ModelConfig = RandomForestConfig | LightGCNConfig | ProfileCoherentLightGCNConfig
