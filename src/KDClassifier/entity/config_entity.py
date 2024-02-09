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
    params_batch_size: int
    params_epochs: int
    params_num_classes: int
    params_learning_rate: float