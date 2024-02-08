from src.KDClassifier import logger
from src.KDClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = "Data Ingestion"
try:
    logger.info(f"###### stage {STAGE_NAME} started ######")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f"###### stage {STAGE_NAME} finished ######")
except Exception as e:
    logger.exception(e)
    raise e