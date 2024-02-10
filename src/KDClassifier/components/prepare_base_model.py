import torch
import torch.nn as nn
from transformers import ViTForImageClassification

from src.KDClassifier import logger
from src.KDClassifier.entity.config_entity import PrepareBaseModelConfig


class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ViTClassifier, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit.classifier = nn.Linear(self.vit.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.vit(pixel_values=x).logits
        return x
    

class PrepareBaseModel:
    '''
    download the model
    '''
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_model(self):
        base_model_path = self.config.base_model_path
        num_classes = self.config.params_num_classes
        logger.info("model creating")
        self.model = ViTClassifier(num_classes)
        self.save_model(path=base_model_path, model=self.model)
        logger.info("model saved")

    
    @staticmethod
    def save_model(path, model):
        torch.save(model, path)
