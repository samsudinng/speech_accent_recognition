from torchvision.datasets import ImageFolder
from torchvision import transforms
from base import BaseDataLoader
import pandas as pd

class AccentDataLoader(BaseDataLoader):
    
    def __init__(self, train_img_path, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
    
        alexnet_transforms = transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.ToTensor(),
                                         #transforms.Lambda(lambda x: torch.unsqueeze(x,1)),
                                         #transforms.Lambda(lambda x: torch.cat((x, x, x), 1)),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                             std=[0.229, 0.224, 0.225])
                                        ]) 

        self.train_img_path = train_img_path
        self.dataset = ImageFolder(root=self.train_img_path, transform=alexnet_transforms)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


"""
class AccentValDataLoader(DataLoader):
    def __init__(self, val_img_path, batch_size, shuffle=False, validation_split=0.0, num_workers=1):

        alexnet_transforms = transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.ToTensor(),
                                         #transforms.Lambda(lambda x: torch.unsqueeze(x,1)),
                                         #transforms.Lambda(lambda x: torch.cat((x, x, x), 1)),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                        ])

        self.val_img_path = val_img_path
        self.dataset = ImageFolder(root=self.val_img_path, transform=alexnet_transforms)
        
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers = num_workers)
"""
