from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True) # simplify the class defination （no need to write def __init__(self)）
class DataIngestionConfig:
    root_dir: Path
    src_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    params_num_classes: int


@dataclass(frozen=True)
class TrainingModelConfig:
    root_dir: Path
    prepare_base_model: Path
    trained_model_path: Path
    training_data: Path
    params_batch_size: int
    params_epochs: int
    params_learning_rate: float


@dataclass(frozen=True)
class TestingConfig:
    model_path: Path
    testing_data: Path
    all_params: dict
    mlflow_uri: str