import os
from PIL import Image
import torch
from torchvision import transforms



class PredictionPipeline:
    def __init__(self, file_name):
        self.file_name = file_name
    
    def predict(self):
        model = torch.load(os.path.join("artifacts", "model_training", "model.pt"))
        model.eval()

        image_name = self.file_name
        image = Image.open(image_name)
        loader = transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor()])
        
        img = loader(image)
        img = img.unsqueeze(0)

        output = model(img)
        pred = torch.argmax(output, 1)

        if pred[0] == 0:
            return(["Normal"])
        else:
            return(["Tumor"])
        

if __name__ == "__main__":
    pred = PredictionPipeline(file_name="C:/Users/15600/Desktop/PY/kidney-disease-classification-project/artifacts/data_ingestion/kidneyCTscan/test/Normal/Normal- (4979).jpg")
    res = pred.predict()
    print(res)

