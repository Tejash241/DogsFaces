from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np 
from glob import glob 
import random
from os.path import join, sep
from PIL import Image, ImageEnhance, ImageOps

class DogDataset(Dataset):    
    def __init__(self, transform=None, sub_folder='train', color='RGB'):        
        self.transform = transform
        self.color = color
        self.image_dir = join('/raid/bart/imagenet2012/', sub_folder)
        self.dog_folders = ['n02110627', 'n02088094', 'n02116738', 'n02096051', 'n02093428', 'n02107908', 'n02096294', 'n02110806', 'n02088238', 'n02088364', 'n02093647', 'n02107683', 'n02089078', 'n02086646', 'n02088466', 'n02088632', 'n02106166', 'n02093754', 'n02090622', 'n02096585', 'n02106382', 'n02108089', 'n02112706', 'n02105251', 'n02101388', 'n02108422', 'n02096177', 'n02113186', 'n02099849', 'n02085620', 'n02112137', 'n02101556', 'n02102318', 'n02106030', 'n02099429', 'n02096437', 'n02115913', 'n02115641', 'n02107142', 'n02089973', 'n02100735', 'n02102040', 'n02108000', 'n02109961', 'n02099267', 'n02108915', 'n02106662', 'n02100236', 'n02097130', 'n02099601', 'n02101006', 'n02109047', 'n02111500', 'n02107574', 'n02105056', 'n02091244', 'n02100877', 'n02093991', 'n02102973', 'n02090721', 'n02091032', 'n02085782', 'n02112350', 'n02105412', 'n02093859', 'n02105505', 'n02104029', 'n02099712', 'n02095570', 'n02111129', 'n02098413', 'n02110063', 'n02105162', 'n02085936', 'n02113978', 'n02107312', 'n02113712', 'n02097047', 'n02111277', 'n02094114', 'n02091467', 'n02094258', 'n02105641', 'n02091635', 'n02086910', 'n02086079', 'n02113023', 'n02112018', 'n02110958', 'n02090379', 'n02087394', 'n02106550', 'n02109525', 'n02091831', 'n02111889', 'n02104365', 'n02097298', 'n02092002', 'n02095889', 'n02105855', 'n02086240', 'n02110185', 'n02097658', 'n02098105', 'n02093256', 'n02113799', 'n02097209', 'n02102480', 'n02108551', 'n02097474','n02113624', 'n02087046', 'n02100583', 'n02089867', 'n02092339', 'n02102177', 'n02098286', 'n02091134', 'n02095314', 'n02094433']
        self.folder_paths = [join(self.image_dir, x) for x in self.dog_folders]
        self.image_filenames = self.get_all_paths()
        print len(self.image_filenames)
        exit()
        self.num_classes = len(self.dog_folders)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image = Image.open(self.image_filenames[idx]).convert(mode=str(self.color))
        if self.transform is not None:
            image = self.transform(image)
            
        # if type(image) is not torch.Tensor:
        #     image = transforms.ToTensor(image)

        # label = self.convert_label(self.dog_folders.index(
        # 							self.image_filenames[idx].split(sep)[-2]), 
        # 								self.num_classes)        
        label = self.dog_folders.index(self.image_filenames[idx].split(sep)[-2])
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

    train_dataset = DogDataset(transform, sub_folder='train')
    val_dataset = DogDataset(transform, sub_folder='val')
    test_dataset = DogDataset(transform, sub_folder='test')

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