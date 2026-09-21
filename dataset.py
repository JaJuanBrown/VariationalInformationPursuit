import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import numpy as np
import pickle
import pandas as pd
from PIL import Image
    
    
def load_mnist(root):
    transform = transforms.Compose([transforms.ToTensor(),  
                                    transforms.Lambda(lambda x: torch.where(x < 0.5, -1., 1.))])
    trainset = datasets.MNIST(root, train=True, transform=transform, download=True)
    testset = datasets.MNIST(root, train=False, transform=transform, download=True)
    return trainset, testset
