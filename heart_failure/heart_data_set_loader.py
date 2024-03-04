import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np 
import math


class HeartDataset(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt('heart_failure/heart_processed.csv', 
                        delimiter=",", dtype=np.float32, skiprows=1)
        num_instances, num_features = xy.shape
        self.x = torch.from_numpy(xy[:, :(num_features - 1)])
        self.y = torch.from_numpy(xy[:, [num_features - 1]]) # n_samples, 1
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
dataset = HeartDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

# training loop
num_epochs = 1
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        if i == 0:  
            #print(inputs, labels)
            print(inputs.shape)

train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

# certainly this is one approach
X_train = [_[0] for _ in train_dataset]
y_train = [_[1] for _ in train_dataset]

X_test = [_[0] for _ in test_dataset]
y_test = [_[1] for _ in test_dataset]

# this is likely a better approach
train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=0)
examples = iter(train_dataloader)
print(next(examples))


