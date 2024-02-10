import os
import torchvision
from torch.utils.data import Dataset


class KidneyDataset(Dataset):
    def __init__(self, root_dir, feature_extractor):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.classes = ['Normal', 'Tumor']
        self.data = []
        for idx, cls in enumerate(self.classes):
            path = os.path.join(root_dir, cls)
            for img in os.listdir(path):
                self.data.append((os.path.join(path, img), idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = torchvision.io.read_image(img_path)
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values[0]
        return image, label