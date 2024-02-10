from src.KDClassifier.config.configuration import ConfigurationManager
from src.KDClassifier.components.training import Training
from src.KDClassifier import logger


STAGE_NAME = "Model Training"

class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(training_config)
        training.get_training_model()
        training.train_valid_generator()
        logger.info("start training")
        training.train()


if __name__ == "__main__":
    try:
        logger.info(f"###### stage {STAGE_NAME} started ######")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f"###### stage {STAGE_NAME} finished ######")
    except Exception as e:
        logger.exception(e)
        raise e