from src.KDClassifier import logger
from src.KDClassifier.config.configuration import ConfigurationManager
from src.KDClassifier.components.testing import Testing


STAGE_NAME = "Model Testing"

class ModelTestingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        test_config = config.get_testing_config()
        testing = Testing(test_config)
        testing.get_testing_model()
        testing.test_generator()
        testing.get_testing()
        testing.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f"###### stage {STAGE_NAME} started ######")
        obj = ModelTestingPipeline()
        obj.main()
        logger.info(f"###### stage {STAGE_NAME} finished ######")
    except Exception as e:
        logger.exception(e)
        raise e
