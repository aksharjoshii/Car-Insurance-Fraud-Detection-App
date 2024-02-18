import os
import pandas as pd
import numpy as np
import torch 
import cv2
from torchvision  import models 
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2



def load_model(predict_config:dict, device:str):
    """
    Load a pre-trained EfficientNet-B0 model with a custom classifier.

    Args:
        config (dict): Configuration dictionary containing model-related parameters.
            It should include 'weights_path', the path to the pre-trained model weights file.
        device (str): Device to which the model should be moved ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: Loaded model with the specified weights and custom classifier.

    Example:
        >>> model_config = {'weights_path': 'path/to/pretrained_model.pth'}
        >>> loaded_model = load_model(model_config, device='cuda')

    """

    model = models.efficientnet_b0(weights=None)
    for param in model.parameters():
        param.requires_grad = False
        model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True),
        torch.nn.Linear(in_features=1280, out_features= 2, bias=True)  
         )
    
    weight_path = predict_config['weights_path']
    model_weights = torch.load(weight_path)
    # Load the weights into the model
    model.load_state_dict(model_weights)

    # Set the model to evaluation mode
    model.eval()
    model.to(device)
    
    return model


def predict_single_image(image_path:str, 
                         model:torch.nn.Module, 
                         device:str):
    """
    Predicts the label and probability for a single image using a loaded model.

    Args:
        image_path (str): Path to the image file.
        model (torch.nn.Module): The pre-trained model for prediction.
        device (str): Device to run the prediction on (e.g., "cpu" or "cuda").

    Returns:
        tuple[str, float]: A tuple containing the predicted label (string)
             and the corresponding probability (float rounded to 4 decimal places).
    """
    # Load the image with error handling
    try:
        image = Image.open(image_path)
    except IOError as e:
        raise IOError(f"Failed to open image: {image_path}, error: {e}")
    
    # Apply transformations
    test_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.Resize(height=224, width=224, interpolation=cv2.INTER_AREA),
        ToTensorV2()
       ])
    
    image_tr = test_transform(image=np.array(image))['image']
    image_tr= torch.tensor(image_tr).unsqueeze(0).to(device)

    # Make predictions
    with torch.no_grad():
        output = model(image_tr)
        probabilities = torch.softmax(output, dim=1)
        _, predicted_class = torch.max(output, 1)
        predicted_class = predicted_class.item()
        if predicted_class == 1:
            prediction = 'Fraudulent'
        else:
            prediction = 'Non-Fraudulent'

    return prediction, round(probabilities[0, predicted_class].item(), 4)




def predict_images_in_dir( model:torch.nn.Module, 
                             device:str,
                             predict_config:dict):
    """
      Predicts labels and probabilities for all images within a specified folder.

    Args:
        folder_path (str): Path to the folder containing the images.
        model (torch.nn.Module): The pre-trained model for prediction.
        device (str): Device to run the predictions on (e.g., "cpu" or "cuda").

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains:
            - 'Filename' (str): The filename of the image.
            - 'Predicted_Label' (str): The predicted label for the image.
            - 'Probability' (float): The probability associated with the prediction.

    Raises:
        OSError: If the specified folder path is invalid.
        RuntimeError: If an error occurs during prediction for any image.
    """
     # Get image paths with error handling
    try:
        image_paths = [os.path.join(predict_config['folder_path'], filename) for filename in os.listdir(predict_config['folder_path'])]
    except OSError as e:
        raise OSError(f"Invalid folder path: {predict_config['folder_path']}, error: {e}")

    # Make predictions
    predictions = []
    for image_path in image_paths:
        try:
            prediction, probability = predict_single_image(image_path, model, device)
            predictions.append({
                "Filename": os.path.basename(image_path),
                "Prediction": prediction,
                "Predict_Probability": probability
            })

        except RuntimeError as e:
            raise RuntimeError(f"Failed to predict image: {image_path}, error: {e}")
    

    df = pd.DataFrame(predictions)
    df.to_csv(predict_config['prediction_path'], index_label=False)


    



