import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

########################################################################

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = CIFAR10(root='data/', download=True, transform=transform)
test_dataset = CIFAR10(root='data/', train=False, transform=transform)

batch_size = 2

classes = dataset.classes
# Build dictionary with counters per-each class
class_count = {}
for _, index in dataset:
    label = classes[index]
    if label not in class_count:
        class_count[label] = 0
    class_count[label] += 1

torch.manual_seed(43)
val_size = 5000    # 10% of the dataset
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# len(train_ds), len(val_ds)

"""
#standardization

print("Prepropcess: [subtract mean], [divide std]")
mean = np.mean(train_ds, axis=(0, 1, 2), keepdims=True)
std = np.std(train_ds, axis=(0, 1, 2), keepdims=True)

print("mean: {}".format(np.reshape(mean * 255.0, [-1])))
print("std: {}".format(np.reshape(std * 255.0, [-1])))
"""

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=2, pin_memory=True)