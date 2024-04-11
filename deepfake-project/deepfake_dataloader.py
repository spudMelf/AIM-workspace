import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd

class DeepfakeDataset(Dataset):
    def __init__(self, directory_path, length=31779, transform=None):
        # Data loading
        df = pd.read_csv(str(directory_path + 'meta.csv'))
        #num_instances, num_features = xy.shape
        self.image_paths = np.array(df['file'])[:length]
        labels = np.array(df['label'])[:length].reshape(-1, 1)
        result = (labels == 'spoof').astype(int)
        self.y = torch.tensor(result)
        self.n_samples = len(labels)
        self.directory_path = directory_path
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.directory_path + str(self.image_paths[index][:-3]) + 'png'
        image = Image.open(image_path).convert('RGB')  # Assuming RGB images
        if self.transform:
            image = self.transform(image)

        label = self.y[index]
        return image, label

    def __len__(self):
        return self.n_samples

dataset = DeepfakeDataset(directory_path='../release_in_the_wild/', transform=transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to the required input size of your CNN
    transforms.ToTensor(),           # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
]))

dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

# test dataloader with image visualization
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(10):
    image, label = dataset[i]
    ax = axes[i // 5, i % 5]
    ax.imshow(image.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C) for plotting
    ax.set_title(f"Label: {label}")
    ax.axis("off")
plt.tight_layout()
plt.show()