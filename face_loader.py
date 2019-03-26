from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np 
from glob import glob 
import random
from os.path import join, sep
from PIL import Image, ImageEnhance, ImageOps
from sklearn.utils import shuffle

class FaceDataset(Dataset):    
    def __init__(self, transform=None, start=True, ratio=1., color='RGB'):        
        self.transform = transform
        self.color = color
        self.image_dir = join('/raid/tvdesai/DogsFaces/LFW_small')
        self.random_seed = 41
        self.ratio = ratio
        self.num_classes = 7
        self.start = start
        self.all_images, self.all_labels = self.get_all_images()

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        image = self.all_images[idx, :]
        image = image.reshape((50, 37))
        image = Image.fromarray(image).convert(mode=str(self.color))
        if self.transform is not None:
            image = self.transform(image)
            
        label = self.all_labels[idx]
        return (image, label)

    def get_all_images(self):
        all_X = np.load(join(self.image_dir, 'X_train.npy'))
        all_y = np.load(join(self.image_dir, 'y_train.npy'))
        all_X, all_y = shuffle(all_X, all_y, random_state=self.random_seed)
        if self.start:
        	return all_X[:int(self.ratio*len(all_X))], all_y[:int(self.ratio*len(all_X))]
        else:
            return all_X[int(self.ratio*len(all_X)):], all_y[int(self.ratio*len(all_X)):]

    def convert_label(self, label, num_classes):
        binary_label = torch.zeros(num_classes, dtype=torch.long)
        binary_label[label] = 1
        return binary_label    
    

def create_loaders(batch_size, transform=None,
                         shuffle=True, extras={}, split=0.92):

    train_dataset = FaceDataset(transform, start=True, ratio=split)
    val_dataset = FaceDataset(transform, start=False, ratio=split)
    print train_dataset.all_images.shape, val_dataset.all_images.shape

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              num_workers=num_workers, 
                              pin_memory=pin_memory, shuffle=shuffle)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_workers, 
                            pin_memory=pin_memory, shuffle=False)
    
    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader)