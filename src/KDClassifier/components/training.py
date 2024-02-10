import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor

from src.KDClassifier import logger
from src.KDClassifier.utils.dataloader import KidneyDataset
from src.KDClassifier.entity.config_entity import TrainingModelConfig


class Training:
    def __init__(self, config: TrainingModelConfig):
        self.config = config

    def get_training_model(self):
        self.model = torch.load(self.config.prepare_base_model)
    
    def train_valid_generator(self):
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

        dataset = KidneyDataset(self.config.training_data, feature_extractor)   

        train_size = int(0.8 * len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

        self.train_loader = DataLoader(train_dataset, batch_size=self.config.params_batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.config.params_batch_size, shuffle=False)

    @staticmethod
    def save_model(model, path):
        torch.save(model, path)

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        logger.info(f"training on {device}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.params_learning_rate)    

        train_losses = []
        train_accuracies = []
        num_epochs = self.config.params_epochs

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)

            logger.info(f'Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}')

            if epoch // 2:
                self.save_model(self.model, self.config.trained_model_path)
                logger.info(f"model saved for epoch {epoch}")