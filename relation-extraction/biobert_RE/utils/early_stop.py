class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, net, patience=3, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        """
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.net = net
    def __call__(self, val_loss, net):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.net = net
        elif self.best_loss > val_loss:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
            # set best model to this one
            self.net = net
        elif self.best_loss < val_loss:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
    
    def get_best_model(self):
        return self.net