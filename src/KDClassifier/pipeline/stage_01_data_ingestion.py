from src.KDClassifier.config.configuration import ConfigurationManager
from src.KDClassifier.components.data_ingestion import DataIngestion
from src.KDClassifier import logger


STAGE_NAME = "Data Ingestion"

class DataIngestionTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()



if __name__ == "__main__":
    try:
        logger.info(f"###### stage {STAGE_NAME} started ######")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f"###### stage {STAGE_NAME} finished ######")
    except Exception as e:
        logger.exception(e)
        raise e
