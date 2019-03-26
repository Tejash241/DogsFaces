from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np 
from glob import glob 
import random
from os.path import join, sep
from PIL import Image, ImageEnhance, ImageOps

class ItemDataset(Dataset):    
    def __init__(self, transform=None, sub_folder='train', color='RGB'):        
        self.transform = transform
        self.color = color
        self.image_dir = join('/raid/bart/imagenet2012/', sub_folder)
        self.item_folders = ['n01687978', 'n01776313', 'n01917289', 'n01924916', 'n01930112', 'n01943899', 'n01945685', 'n02264363', 'n02268443', 'n02268853', 'n02321529', 'n02526121', 'n02669723', 'n02672831', 'n02676566', 'n02687172', 'n02690373', 'n02692877', 'n02699494', 'n02701002', 'n02708093', 'n02727426', 'n02730930', 'n02747177', 'n02749479', 'n02769748', 'n02776631', 'n02777292', 'n02782093', 'n02783161', 'n02786058', 'n02787622', 'n02788148', 'n02790996', 'n02791124', 'n02791270', 'n02793495', 'n02794156', 'n02795169', 'n02797295', 'n02799071', 'n02802426', 'n02804414', 'n02804610', 'n02807133', 'n02808304', 'n02808440', 'n02814533', 'n02814860', 'n02837789', 'n02870880', 'n02892767', 'n02917067', 'n02948072', 'n02965783', 'n02966193', 'n02966687', 'n02971356', 'n02974003', 'n02977058', 'n02978881', 'n02979186', 'n02980441', 'n02981792', 'n02988304', 'n02992211', 'n02992529', 'n02999410', 'n03000134', 'n03000247', 'n03000684', 'n03014705', 'n03016953', 'n03017168', 'n03018349', 'n03026506', 'n03028079', 'n03032252', 'n03041632', 'n03045698', 'n03047690', 'n03062245', 'n03063599', 'n03063689', 'n03065424', 'n03075370', 'n03085013', 'n03089624', 'n03095699', 'n03100240', 'n03109150', 'n03110669', 'n03125729', 'n03126707', 'n03133878', 'n03146219', 'n03179701', 'n03187595', 'n03197337', 'n03207743', 'n03207941', 'n03216828', 'n03223299', 'n03240683', 'n03249569', 'n03250847', 'n03255030', 'n03259280', 'n03272010', 'n03272562', 'n03291819', 'n03314780', 'n03344393', 'n03345487', 'n03355925', 'n03388043', 'n03394916', 'n03424325', 'n03444034', 'n03445924']
        self.folder_paths = [join(self.image_dir, x) for x in self.item_folders]
        self.image_filenames = self.get_all_paths()
        print len(self.image_filenames)
        exit()
        self.num_classes = len(self.item_folders)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image = Image.open(self.image_filenames[idx]).convert(mode=str(self.color))
        if self.transform is not None:
            image = self.transform(image)
            
        # if type(image) is not torch.Tensor:
        #     image = transforms.ToTensor(image)

        # label = self.convert_label(self.item_folders.index(
        # 							self.image_filenames[idx].split(sep)[-2]), 
        # 								self.num_classes)        
        label = self.item_folders.index(self.image_filenames[idx].split(sep)[-2])
        return (image, label)

    def get_all_paths(self):
    	all_paths = []
    	for folder in self.folder_paths:
    		all_paths.extend(glob(join(folder, '*.JPEG')))

    	return all_paths

    def convert_label(self, label, num_classes):
        binary_label = torch.zeros(num_classes, dtype=torch.long)
        binary_label[label] = 1
        return binary_label    
    

def create_loaders(batch_size, transform=None,
                         shuffle=True, extras={}):

    train_dataset = ItemDataset(transform, sub_folder='train')
    val_dataset = ItemDataset(transform, sub_folder='val')
    test_dataset = ItemDataset(transform, sub_folder='test')

    if shuffle:
        np.random.shuffle(train_dataset.image_filenames)    

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              num_workers=num_workers, 
                              pin_memory=pin_memory, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_workers, 
                            pin_memory=pin_memory, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                              num_workers=num_workers, 
                              pin_memory=pin_memory, shuffle=False)
    
    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader, test_loader)