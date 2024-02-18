import torch
from pipelines.prediction_pipeline import *

predict_config = {
    'weights_path' : 'artifacts/best_model.pth',
    'folder_path' : 'data_dir/test/images',
    'prediction_path' : 'data_dir/test/predictions.csv',


}
device = "cuda" if torch.cuda.is_available() else "cpu"
if __name__ == "__main__":
    model = load_model(device=device, predict_config=predict_config)
    predict_images_in_dir(model=model, device=device, predict_config=predict_config)
    print("Done")