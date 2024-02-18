from copy import deepcopy



class EarlyStopping:
    """Custom Early Stopping class with enhancements."""

    def __init__(self, patience=3, min_delta=0.05, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None  # Store a backup of the best model's state
        self.best_f1 = float('-inf')  # Track the best F1-score achieved
        self.counter = 0  # Patience counter
        self.status = ""  # Store status messages for logging

    def __call__(self, model, val_f1):
        """Early stopping logic during validation loop."""
        if self.best_f1 is None or val_f1 - self.best_f1 >= self.min_delta:
            # Improvement found: update best model, reset counter, log message
            self.best_model = deepcopy(model.state_dict())
            self.best_f1 = val_f1
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            # No improvement: increment counter, log message
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                # Early stopping triggered: restore best model if needed, log message
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True  # Indicate early stopping
        return False  # Continue training





# class EarlyStopping:
#     """Custom Early Stopping class with enhancements."""

#     def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.restore_best_weights = restore_best_weights
#         self.best_model = None  # Store a backup of the best model's state
#         self.best_loss = None  # Track the best loss achieved
#         self.counter = 0  # Patience counter
#         self.status = ""  # Store status messages for logging

#     def __call__(self, model, val_loss):
#         """Early stopping logic during validation loop."""
#         if self.best_loss is None or val_loss - self.best_loss >= self.min_delta:
#             # Improvement found: update best model, reset counter, log message
#             self.best_model = deepcopy(model.state_dict())
#             self.best_loss = val_loss
#             self.counter = 0
#             self.status = f"Improvement found, counter reset to {self.counter}"
#         else:
#             # No improvement: increment counter, log message
#             self.counter += 1
#             self.status = f"No improvement in the last {self.counter} epochs"
#             if self.counter >= self.patience:
#                 # Early stopping triggered: restore best model if needed, log message
#                 self.status = f"Early stopping triggered after {self.counter} epochs."
#                 if self.restore_best_weights:
#                     model.load_state_dict(self.best_model)
#                 return True  # Indicate early stopping
#         return False  # Continue training

# class EarlyStopping:
#     def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.restore_best_weights = restore_best_weights
#         self.best_model = None
#         self.best_loss = None
#         self.counter = 0
#         self.status = ""

#     def __call__(self, model, val_loss):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#             self.best_model = copy.deepcopy(model.state_dict())
#         elif self.best_loss - val_loss >= self.min_delta:
#             self.best_model = copy.deepcopy(model.state_dict())
#             self.best_loss = val_loss
#             self.counter = 0
#             self.status = f"Improvement found, counter reset to {self.counter}"
#         else:
#             self.counter += 1
#             self.status = f"No improvement in the last {self.counter} epochs"
#             if self.counter >= self.patience:
#                 self.status = f"Early stopping triggered after {self.counter} epochs."
#                 if self.restore_best_weights:
#                     model.load_state_dict(self.best_model)
#                 return True
#         return False