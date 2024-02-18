import os
from KDClassifier.constants import *
from KDClassifier.utils.common import read_yaml, create_directories
from KDClassifier.entity.config_entity import (DataIngestionConfig, 
                                                   PrepareBaseModelConfig,
                                                   TrainingModelConfig,
                                                   TestingConfig)


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
            params_num_classes=self.params.NUM_CLASSES
        
        )
        return base_model_config
    
    def get_training_config(self) -> TrainingModelConfig:
        prepare_base_model = self.config.prepare_base_model
        model_training = self.config.model_training
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "train")
        create_directories([model_training.root_dir])

        training_config = TrainingModelConfig(
            root_dir=Path(model_training.root_dir),
            prepare_base_model=Path(prepare_base_model.base_model_path),
            trained_model_path=Path(model_training.trained_model_path), # for saving
            training_data=Path(training_data),
            params_batch_size=self.params.BATCH_SIZE,
            params_epochs=self.params.EPOCHS,
            params_learning_rate=self.params.LEARNING_RATE

        )
        return training_config
    
    def get_testing_config(self) -> TestingConfig:
        testing_data = os.path.join(self.config.data_ingestion.unzip_dir, "test")
        test_config = TestingConfig(
            model_path=Path(self.config.model_training.trained_model_path),
            testing_data=Path(testing_data),
            all_params=self.params,
            mlflow_uri=str(self.config.model_testing.mlflow_uri)

        )
        return test_config