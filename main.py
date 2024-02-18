import os
from KDClassifier import logger
from KDClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from KDClassifier.pipeline.stage_02_model_prepare import ModelPreparePipeline
from KDClassifier.pipeline.stage_03_training import ModelTrainingPipeline
from KDClassifier.pipeline.stage_04_testing import ModelTestingPipeline


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
    obj = ModelPreparePipeline()
    obj.main()
    logger.info(f"###### stage {STAGE_NAME} finished ######")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Training"
try:
    logger.info(f"###### stage {STAGE_NAME} started ######")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f"###### stage {STAGE_NAME} finished ######")
except Exception as e:
    logger.exception(e)
    raise e


######  PUT THESE PARAMS INTO secret_config.yaml
os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/yhr0864/kidney-disease-classification-project.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="yhr0864"
os.environ["MLFLOW_TRACKING_PASSWORD"]="d9d7c85432b94cdcba7ac783932b5090561e756d"


STAGE_NAME = "Model Testing"
try:
    logger.info(f"###### stage {STAGE_NAME} started ######")
    obj = ModelTestingPipeline()
    obj.main()
    logger.info(f"###### stage {STAGE_NAME} finished ######")
except Exception as e:
    logger.exception(e)
    raise e