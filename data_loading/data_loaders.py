from torch.utils.data import DataLoader, WeightedRandomSampler
from .image_transforms import get_transforms
from .custom_data import CustomDataset

# Function to create DataLoaders  
def get_dataloaders(config,
                    root_dir: str,
                    ):
    """
    
    """
    minority_transform, majority_transform, test_transform = get_transforms()
    train_dataset = CustomDataset(data_dir=root_dir, mode='train', transform={'minority': minority_transform, 'majority': majority_transform})
    valid_dataset = CustomDataset(data_dir=root_dir, mode='valid', transform=test_transform)
    

    if config.sampling : 
        # Create WeightedRandomSampler for training dataset
        weights_train = [train_dataset.class_weights[label] for label in train_dataset.labels]
        sampler_train = WeightedRandomSampler(weights_train, len(weights_train), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler= sampler_train, num_workers=8, pin_memory=True)
        validation_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers= 8, pin_memory=True)
    else: 
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        validation_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, validation_loader



