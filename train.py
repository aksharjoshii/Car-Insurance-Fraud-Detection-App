from pipelines.training_pipeline import model_pipeline
from utils.utils import EarlyStopping
import  torch 
import wandb
import os 

# device agnostic interface
data_dir = 'data_dir/car_images'
device = "cuda" if torch.cuda.is_available() else "cpu"
# early_stopping = EarlyStopping(patience=5 , min_delta=0.05, restore_best_weights=True )
# wandb project/run  config
wandb_proj = {
    "project": "Insurance_Fraud_Final",
    "name": "efnet0_b16_e15_WRsampling_repalce",
} 

config = dict(
    epochs = 10,
    classes = 2, 
    batch_size = 16,
    lr = 0.001,
    architecture = 'EfficientNet_B0',
    #num_workers = os.cpu_count(),
    replacement = True,
    sampling = True,  
    dropout = 0.3, 
    best_weights = 'best_weights_v2.pth'
)


if __name__ == '__main__':
    wandb.login()
    model_pipeline(proj_config=wandb_proj, hyperparams=config, data_dir=data_dir, device=device)
    



    





    
    

