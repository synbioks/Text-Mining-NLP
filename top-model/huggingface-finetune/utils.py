from transformers import TrainerCallback
import json
import matplotlib.pyplot as plt

class LogCallback(TrainerCallback):
    """
    A bare :class:`~transformers.TrainerCallback` that just prints the logs.
    """
    def __init__(self, logs_file) -> None:
        self.log_file = logs_file
        self.logs = []


    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            self.logs.append(logs)

    def on_train_end(self, args, state, control, **kwargs):
        with open(self.log_file,'w') as fp:
            json.dump(self.logs,fp)

def plot_loss_log(log_file):
    '''
    This is how each line look like
    {'loss': 39.7785, 'learning_rate': 0.00045000000000000004, 'epoch': 5.0}
    {'eval_loss': 31.266027450561523, 'eval_runtime': 3.1009, 'eval_samples_per_second': 55.468, 'epoch': 5.0}
    '''
    with open(log_file,'r') as fp:
        lines = json.load(fp)
    train_loss = {}
    val_loss = {}
    for line in lines:
        if 'loss' in line.keys():
            train_loss[line['epoch']]=line['loss']
        if 'eval_loss' in line.keys():
            val_loss[line['epoch']]=line['eval_loss']
    epochs_t  = list(train_loss.keys())
    loss_t = list(train_loss.values())
    plt.plot(epochs_t, loss_t, 'o-', label="Train Loss")
    epochs_v  = list(val_loss.keys())
    loss_v = list(val_loss.values())
    plt.plot(epochs_v, loss_v, 'o-', label="Val Loss")
    plt.legend()
    plt.savefig(log_file.split('.')[0]+'.png')
    


# if __name__ == "__main__":
#     a = [
#         {'loss': 39.7785, 'learning_rate': 0.00045000000000000004, 'epoch': 5.0},
#         {'loss': 35.7785, 'learning_rate': 0.00045000000000000004, 'epoch': 10.0},
#         {'eval_loss': 31.266027450561523, 'eval_runtime': 3.1009, 'eval_samples_per_second': 55.468, 'epoch': 5.0}]
#     with open('test.json','w') as fp:
#         json.dump(a,fp)
#     plot_loss_log('test.json')

