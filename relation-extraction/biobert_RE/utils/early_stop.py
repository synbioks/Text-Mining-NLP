class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, net, patience=6, min_delta=0.0001):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.net = net
        self.step_count = 0
    def __call__(self, val_loss, net, step_count):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.net = net
            self.step_count = step_count
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
            # set best model to this one
            self.net = net
            # set the best step count to this one
            self.step_count = step_count
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
    
    def get_best_model(self):
        return [self.net, self.step_count]