import torch
import pandas as pd
import skimage.io as sk
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
# Need to override __init__, __len__, __getitem__
# as per datasets requirement
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, labelsFile, rootDir, sourceTransform):
        self.data = pd.read_csv(labelsFile)
        self.rootDir = rootDir
        self.sourceTransform = sourceTransform
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        print("getitem")
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imagePath = self.rootDir + "/" + self.data['Image_path'][idx]
        image = sk.imread(imagePath)
        label = self.data['Condition'][idx]
        image = Image.fromarray(image)

        if self.sourceTransform:
            image = self.sourceTransform(image)

        return image, label