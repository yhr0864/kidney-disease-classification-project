from src.KDClassifier.config.configuration import ConfigurationManager
from src.KDClassifier.components.prepare_base_model import PrepareBaseModel 
from src.KDClassifier import logger


STAGE_NAME = "Model Prepare"

class ModelPrepareTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        PrepareBaseModelConfig = config.get_base_model_config()
        prepare_base_model = PrepareBaseModel(config=PrepareBaseModelConfig)
        batch_size, epochs, learning_rate = prepare_base_model.get_base_model()

        return batch_size, epochs, learning_rate


if __name__ == "__main__":
    try:
        logger.info(f"###### stage {STAGE_NAME} started ######")
        obj = ModelPrepareTrainingPipeline()
        batch_size, epochs, learning_rate = obj.main()
        logger.info(f"###### stage {STAGE_NAME} finished ######")
    except Exception as e:
        logger.exception(e)
        raise e
