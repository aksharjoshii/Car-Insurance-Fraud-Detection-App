import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_transforms():
    """
    Returns data augmentation transforms for minority class, majority class, and testing.

    Returns:
    - minority_transform (albumentations.Compose): Augmentation transforms for minority class.
    - majority_transform (albumentations.Compose): Augmentation transforms for majority class.
    - test_transform (albumentations.Compose): Transformations for testing without augmentation.
    
    The transforms include horizontal flip, shift-scale-rotate, random rotate 90 degrees, optical distortion,
    CLAHE, random brightness and contrast adjustments, normalization, resizing, and tensor conversion.
    
    """
    minority_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomRotate90(p=0.3),
        A.OpticalDistortion(always_apply=False, p=0.4, distort_limit=(-0.23, 0.09), shift_limit=(-0.13, 0.22), interpolation=1),
        A.CLAHE(always_apply=False, p=0.3, clip_limit=(1, 11), tile_grid_size=(5, 5)),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.Resize(height=224, width=224, interpolation=cv2.INTER_AREA), 
        ToTensorV2()
    ])
    majority_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.Resize(height=224, width=224, interpolation=cv2.INTER_AREA), 
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.Resize(height=224, width=224, interpolation=cv2.INTER_AREA),
        ToTensorV2()
        
    ])

    return minority_transform, majority_transform, test_transform
