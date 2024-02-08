from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True) # simplify the class defination （no need to write def __init__(self)）
class DataIngestionConfig:
    root_dir: Path
    src_URL: str
    local_data_file: Path
    unzip_dir: Path