import tqdm
import torch
import mlflow
from urllib.parse import urlparse
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, precision_recall_fscore_support)

from KDClassifier import logger
from KDClassifier.utils.dataloader import KidneyDataset
from KDClassifier.entity.config_entity import TestingConfig


class Testing:
    def __init__(self, config: TestingConfig):
        self.config = config
    
    def get_testing_model(self):
        self.model = torch.load(self.config.model_path)

    def test_generator(self):
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

        test_dataset = KidneyDataset(self.config.testing_data, feature_extractor)   
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    def get_testing(self):
        self.model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for (inputs, labels) in tqdm.tqdm(self.test_loader,
                                              total=len(self.test_loader)):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        self.accuracy = accuracy_score(all_labels, all_preds)
        self.precision, self.recall, self.f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        logger.info(f"Accuracy: {self.accuracy}\nPrecision: {self.precision}\nRecall: {self.recall}\nF1 Score: {self.f1}")
        logger.info(classification_report(all_labels, all_preds, target_names=['Normal', 'Tumor']))

        # Confusion Matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        logger.info(f"conf_matrix:\n {conf_matrix}")
    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"Accuracy": self.accuracy,
                                "Precision": self.precision,
                                "Recall": self.recall,
                                "F1": self.f1

            })
            # model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.pytorch.log_model(self.model, "model", registered_model_name="ViT")
            else:
                mlflow.pytorch.log_model(self.model, "model")