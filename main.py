from src.KDClassifier import logger
from src.KDClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.KDClassifier.pipeline.stage_02_model_prepare import ModelPrepareTrainingPipeline


STAGE_NAME = "Data Ingestion"
try:
    logger.info(f"###### stage {STAGE_NAME} started ######")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f"###### stage {STAGE_NAME} finished ######")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Prepare"
try:
    logger.info(f"###### stage {STAGE_NAME} started ######")
    obj = ModelPrepareTrainingPipeline()
    batch_size, epochs, learning_rate = obj.main()
    logger.info(f"###### stage {STAGE_NAME} finished ######")
except Exception as e:
    logger.exception(e)
    raise e