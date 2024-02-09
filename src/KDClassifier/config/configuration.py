from src.KDClassifier.constants import *
from src.KDClassifier.utils.common import read_yaml, create_directories
from src.KDClassifier.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig


class ConfigurationManager:
    '''
    read the config.yaml
    return data_ingestion_config
    '''
    def __init__(self, config_filepath=CONFIG_FILE_PATH,
                       params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            src_URL=config.src_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config
    
    def get_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])

        base_model_config = PrepareBaseModelConfig(
            root_dir=config.root_dir,
            base_model_path=config.base_model_path,
            params_batch_size=self.params.BATCH_SIZE,
            params_epochs=self.params.EPOCHS,
            params_num_classes=self.params.NUM_CLASSES,
            params_learning_rate=self.params.LEARNING_RATE

        )
        return base_model_config