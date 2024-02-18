from KDClassifier import logger
from KDClassifier.config.configuration import ConfigurationManager
from KDClassifier.components.prepare_base_model import PrepareBaseModel 


STAGE_NAME = "Model Prepare"

class ModelPreparePipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        PrepareBaseModelConfig = config.get_base_model_config()
        prepare_base_model = PrepareBaseModel(config=PrepareBaseModelConfig)
        prepare_base_model.get_base_model()


if __name__ == "__main__":
    try:
        logger.info(f"###### stage {STAGE_NAME} started ######")
        obj = ModelPreparePipeline()
        obj.main()
        logger.info(f"###### stage {STAGE_NAME} finished ######")
    except Exception as e:
        logger.exception(e)
        raise e
