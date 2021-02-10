from torchvision.datasets import ImageFolder
from torchvision import transforms
from base import BaseDataLoader
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Any, Tuple
import numpy as np
from torchaudio import transforms as audiotransforms
from facenet_pytorch import fixed_image_standardization

class ImageFolderAlbumentations(ImageFolder):
    def __init__(self, root='', transform=None):
        super(ImageFolderAlbumentations, self).__init__(root=root, transform=transform)
        print(self.loader)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            transformed = self.transform(image=np.asarray(sample))
            sample = transformed['image']
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class AccentTrainDataLoader(BaseDataLoader):
    
    def __init__(self, train_img_path, batch_size, shuffle=True, validation_split=0.0, num_workers=1, p_augment=0.0, training=True):
        
        

        alexnet_transforms = transforms.Compose([
                                        transforms.Resize(256),
                                         transforms.ToTensor(), 
                                        transforms.RandomChoice([
                                            transforms.RandomApply([audiotransforms.FrequencyMasking(freq_mask_param=50)], p=p_augment),
                                            transforms.RandomApply([audiotransforms.TimeMasking(time_mask_param=100)], p=p_augment)
                                                                 ]),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])

                                        ])
        
        """
        alexnet_transforms = A.Compose([
                                         A.Resize(256,256,always_apply=True),
                                         A.OneOf([
                                            A.GaussNoise(p=0.4),
                                            A.RandomBrightnessContrast(p=0.4),
                                            A.ShiftScaleRotate(shift_limit_x = 0.1,scale_limit = 0, rotate_limit=0,p=0.4)
                                        ], p = p_augment),
                                         A.Normalize(mean=[0.485, 0.456, 0.406], 
                                                             std=[0.229, 0.224, 0.225],always_apply=True),
                                         ToTensorV2(always_apply=True)
                                        ]) 
       
        alexnet_transforms = transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.ToTensor(),
                                         #transforms.Lambda(lambda x: torch.unsqueeze(x,1)),
                                         #transforms.Lambda(lambda x: torch.cat((x, x, x), 1)),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                        ])
        """

        self.train_img_path = train_img_path
        #self.dataset = ImageFolderAlbumentations(root=self.train_img_path, transform=alexnet_transforms)
        self.dataset = ImageFolder(root=self.train_img_path, transform=alexnet_transforms)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class AccentDataLoader(BaseDataLoader):

    def __init__(self, img_path, batch_size, shuffle=False, num_workers=1,validation_split=0.0):



        alexnet_transforms = transforms.Compose([
                                        transforms.Resize(256),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])

                                        ])

        self.img_path = img_path
        self.dataset = ImageFolder(root=self.img_path, transform=alexnet_transforms)
        super().__init__(self.dataset, batch_size, shuffle,validation_split, num_workers)

"""
class AccentValDataLoader(DataLoader):
    def __init__(self, val_img_path, batch_size, shuffle=False, validation_split=0.0, num_workers=1):
        
        alexnet_transforms = A.Compose([
                                         A.Resize(256,256,always_apply=True),
                                         A.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225],always_apply=True),
                                         ToTensorV2(always_apply=True)
                                        ])

        
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



class AccentInceptionResnetV1TrainDataLoader(BaseDataLoader):

    def __init__(self, train_img_path, batch_size, shuffle=True, validation_split=0.0, num_workers=1, p_augment=0.0, training=True):



        resnet50_transforms =transforms.Compose([
                                        transforms.Resize((160,160)),
                                        transforms.ToTensor(),
                                        transforms.RandomChoice([
                                            transforms.RandomApply([audiotransforms.FrequencyMasking(freq_mask_param=50)], p=p_augment),
                                            transforms.RandomApply([audiotransforms.TimeMasking(time_mask_param=100)], p=p_augment)
                                                                 ]),
                                        fixed_image_standardization
                            ])

        self.train_img_path = train_img_path
        self.dataset = ImageFolder(root=self.train_img_path, transform=resnet50_transforms)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    
class AccentInceptionResnetV1DataLoader(BaseDataLoader):

    def __init__(self, img_path, batch_size, shuffle=False, num_workers=1,validation_split=0.0):


        resnet50_transforms =transforms.Compose([
                                        transforms.Resize((160,160)),
                                        transforms.ToTensor(),
                                        fixed_image_standardization
                            ])


        self.img_path = img_path
        self.dataset = ImageFolder(root=self.img_path, transform=resnet50_transforms)
        super().__init__(self.dataset, batch_size, shuffle,validation_split, num_workers)




class AccentVGGMTrainDataLoader(BaseDataLoader):

    def __init__(self, train_img_path, batch_size, shuffle=True, validation_split=0.0, num_workers=1, p_augment=0.0, training=True):



        img_transforms = transforms.Compose([
                                        transforms.Resize((512,300)),
                                         transforms.ToTensor(),
                                        transforms.RandomChoice([
                                            transforms.RandomApply([audiotransforms.FrequencyMasking(freq_mask_param=50)], p=p_augment),
                                            transforms.RandomApply([audiotransforms.TimeMasking(time_mask_param=100)], p=p_augment)
                                                                 ]),
                                        ])


        self.train_img_path = train_img_path
        self.dataset = ImageFolder(root=self.train_img_path, transform=img_transforms)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class AccentVGGMDataLoader(BaseDataLoader):

    def __init__(self, img_path, batch_size, shuffle=True, validation_split=0.0, num_workers=1, p_augment=0.0, training=True):



        img_transforms = transforms.Compose([
                                        transforms.Resize((512,300)),
                                         transforms.ToTensor(),
                                        ])


        self.img_path = img_path
        self.dataset = ImageFolder(root=self.img_path, transform=img_transforms)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

