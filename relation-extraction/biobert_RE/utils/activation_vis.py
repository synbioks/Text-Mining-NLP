import os
import numpy as np
import matplotlib.pyplot as plt

class ActivationHook:

    def __init__(self, name):
        self.name = name
        self.activations = []

    def __call__(self, net, x, y):
        self.activations.append(y.detach().cpu().numpy())

    def get_activation(self):
        res = []
        for batch in self.activations:
            for item in batch:
                res.append(item)
        return np.array(res)

    def vis_activation(self, save_directory=None):
        data = self.get_activation()
        data = np.sort(np.var(data, axis=0))
        width = len(data)
        if width > 16:
            data = data.reshape(16, width // 16)
        else:
            data = data.reshape(1, width)
        data_max = np.amax(data)
        data_min = np.amin(data)
        data_range = data_max - data_min
        # print(f"min: {data_min}, max: {data_max}")
        data = (data - data_min) / data_range
        fig, ax = plt.subplots()
        im = ax.imshow(data)
        fig.colorbar(im)
        if save_directory is None:
            plt.show()
        else:
            plt.savefig(os.path.join(save_directory, f'{self.name}.png'))