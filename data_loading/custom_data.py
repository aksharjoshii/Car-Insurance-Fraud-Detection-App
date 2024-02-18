import os 
import cv2
from torch.utils.data import  Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dir, mode='train', transform=None):
        """
        CustomDataset constructor.

        Parameters:
            - data_dir (str): Path to the root directory containing data.
            - mode (str): Dataset mode ('train', 'validation', or 'test').
            - transform (dict): Dictionary containing transformations for minority and majority classes.
        """
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        # Get image paths and labels based on your specific folder structure
        self.img_paths, self.labels = self._get_data_from_folders(mode)

        # Calculate class weights
        self.class_weights = self._calculate_class_weights()

    def _get_data_from_folders(self, mode):
        """
        Internal method to retrieve image paths and labels from folders.

        Parameters:
            - mode (str): Dataset mode ('train', 'validation', or 'test').

        Returns:
            - img_paths (list): List of image file paths.
            - labels (list): List of corresponding class labels.
        """
        img_paths = []
        labels = []
        data_path = os.path.join(self.data_dir, mode)

        # Assuming subfolders 'fraudulent' and 'non-fraudulent' for labels
        for label_dir in os.listdir(data_path):
            label_path = os.path.join(data_path, label_dir)
            # Map label string to numerical value (adjust based on your labels)
            label = {"fraudulent": 1, "non-fraudulent": 0}[label_dir]
            for img_filename in os.listdir(label_path):
                img_path = os.path.join(label_path, img_filename)
                img_paths.append(img_path)
                labels.append(label)

        return img_paths, labels

    def _calculate_class_weights(self):
        """
        Internal method to calculate class weights based on class frequencies.

        Returns:
            - class_weights (dict): Dictionary containing class weights.
        """
        class_freqs = {}
        for label in self.labels:
            if label not in class_freqs:
                class_freqs[label] = 0
            class_freqs[label] += 1

        total_data = len(self.labels)
        class_weights = {label: total_data / (class_freqs[label] * len(class_freqs)) for label in class_freqs}
        return class_weights

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            - len (int): Number of samples.
        """
        return len(self.img_paths)

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset.

        Parameters:
            - index (int): Index of the sample to retrieve.

        Returns:
            - img (ndarray): Image data.
            - label (int): Class label.
        """
        img_path = self.img_paths[index]
        label = self.labels[index]

        img = self._load_image(img_path)

        if self.transform is not None:
            if self.mode == 'train':
                # Apply different transformations based on minority/majority class during training
                transform_to_apply = self.transform['minority'] if label == 1 else self.transform['majority']
                img = transform_to_apply(image=img)
            else:
                # Apply the same transformation for both classes during validation
                img = self.transform(image=img)
            
        # Return img and label for evaluation or img, label, weight for training
        if self.mode == 'train':
            return img, label, self.class_weights[label]
        else:
            return img, label

    def _load_image(self, img_path):
        """
        Internal method to load an image from the given path.

        Parameters:
            - img_path (str): Path to the image file.

        Returns:
            - img (ndarray): Loaded image data.
        """
        # Handle different image formats and conversions
        img = cv2.imread(img_path)
        # ... additional processing/conversions as needed
        return img
