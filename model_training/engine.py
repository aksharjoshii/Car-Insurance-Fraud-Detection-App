import torch 
import wandb
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

def training_model(model,dataloader,
                   optimizer, 
                   loss_fn, 
                   device
                    ):
    
    #wandb.watch(model, loss_fn, log='all', log_freq='25')

    model.train()
    L , acc = 0, 0
    true_labels , predicted_labels = [], [] 
    for i , (images, labels, _) in enumerate(dataloader):

        images = images['image'].to(device)
        labels = labels.to(device)
       
        # Forward Pass
        y_pred = model(images)
        loss = loss_fn(y_pred, labels)
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        #Step with optimizer
        optimizer.step()
        
        L += loss.item()
        acc += (y_pred.argmax(dim=1) == labels).sum().item()

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(y_pred.argmax(dim=1).cpu().numpy())
    
    # Calculate average loss and accuracy
    avg_loss = L / len(dataloader)
    accuracy = acc/ (len(true_labels))
    
    # Calculate precision, recall, and AUC-PR
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    cm = confusion_matrix(true_labels, predicted_labels)
    

    training_results = {
        'Loss' : avg_loss,
        'Accuracy' : accuracy,
        'Precision' : precision,
        'Recall' : recall,
        'F1-Macro' : f1,
        'Confusion Matrix' : cm
    }
    
    
    return training_results



def validation_model(model,
                     dataloader,
                     loss_fn, 
                     device,
                     config
                     ):

    model.eval()
    L , acc = 0, 0
    true_labels , predicted_labels, pred_probas = [], [], []
    
    with torch.no_grad():
    
        for i , (images, labels) in enumerate(dataloader):
            images = images['image'].to(device)
            labels = labels.to(device)
          
            y_pred = model(images)
            loss = loss_fn(y_pred, labels)
            L += loss.item()
            acc += (y_pred.argmax(dim=1) == labels).sum().item()

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(y_pred.argmax(dim=1).detach().cpu().numpy())
            pred_probas.extend(torch.nn.functional.softmax(y_pred, dim=1).cpu().numpy())


    
    # Calculate average loss and accuracy
    avg_loss = L / len(dataloader)
    accuracy = acc/ (len(true_labels))
    
    # Calculate precision, recall, and AUC-PR
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    cm = confusion_matrix(true_labels, predicted_labels)
    
   

    validation_results = {
        'Loss' : avg_loss,
        'Accuracy' : accuracy,
        'Precision' : precision,
        'Recall' : recall,
        'F1-Macro' : f1,
        'Confusion Matrix' : cm
    }

    #  # Early stopping check based on F1-score
    # if config.early_stopping and config.early_stopping(model, f1):
    #     print("Early stopping triggered.")
    #     model.train()
    #     return None  # Indicate early stopping


    model.train()
        
    return true_labels, pred_probas, validation_results







