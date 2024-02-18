import torch 
import torch.nn as nn
from tqdm.auto import tqdm
import wandb
import json
from data_loading.data_loaders import get_dataloaders
from model_training.engine import training_model, validation_model
from model_training.create_model import get_model_optimizer

# setup a device 
device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = 'data_dir/car_images'
best_metrics_path = 'artifacts/best_metrics.json'
def model_pipeline( hyperparams:dict,proj_config:dict, data_dir=data_dir, device=device):
        wandb.init(**proj_config, config=hyperparams)
        config = wandb.config

        # get  data loaders 
        train_loader, valid_loader = get_dataloaders(config=config, root_dir=data_dir)
        # get model , optimisers and loss function 
        model, optimizer = get_model_optimizer(config=config, device=device)
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()

        best_val_f1 = float('-inf')
        best_model_state_dict = None
        for epoch in tqdm(range(config.epochs)):

        
            train_res = training_model(model=model, 
                                            dataloader=train_loader,
                                            optimizer=optimizer,
                                            loss_fn=loss_fn,
                                            device=device)
            

            y_true, y_probas, val_res   = validation_model(model=model, 
                                       dataloader=valid_loader,
                                       loss_fn=loss_fn,
                                       config=config,
                                       device=device)
            
            if val_res['F1-Macro'] > best_val_f1:
                best_val_f1 = val_res['F1-Macro']
                best_model_state_dict = model.state_dict().copy()
                
                # Save best evaluation metrics as JSON
            best_metrics = {
                'epoch': epoch + 1,
                'Loss': val_res['Loss'],
                'Precision': val_res['Precision'],
                'Recall': val_res['Recall'],
                'F1-Macro': val_res['F1-Macro'],

            }
            with open(best_metrics_path, 'w') as json_file:
                json.dump(best_metrics, json_file)

            wandb.log({'train/loss':train_res['Loss'] ,'train/precsion': train_res['Precision'],
                       'train/recall':train_res['Recall'] ,'train/f1-macro':train_res['F1-Macro'],
                       })
            
            wandb.log({'valid/loss':val_res['Loss'],'valid/precsion': val_res['Precision'],
                       'valid/recall':val_res['Recall'],'valid/f1-macro':val_res['F1-Macro'],
                       'valid/roc_curve':wandb.plot.roc_curve(y_true=y_true, 
                                                              y_probas=y_probas,
                                                              labels=['non-fraudulent', 'fraudulent'])
                       })
            
        
            print("**" * 30)
            print("EPOCH :", epoch+1)
            print('Training Results: ', train_res)
            print('validation_results: ', val_res)
            print("**" * 30)
            

        # Save the best model after training loop
        if best_model_state_dict is not None:
            torch.save(best_model_state_dict, 'artifacts/best_model.pth')
            print("Saved best model with F1-macro:", best_val_f1)    
        
        # Create and log a W&B artifact for the model
        model_artifact = wandb.Artifact("trained_model", type="model", description="EfnetB0 for Insurance Fraud Detection")
        model_artifact.add_file('artifacts/best_model.pth')
        wandb.log_artifact(model_artifact)
            
        wandb.finish()

        


        






        