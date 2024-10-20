import torch
import inspect

def add_to_class(Class):
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)

    return wrapper

class Hyperparameters:
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented

    def save_hyperparameters(self, ignore=[]):
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)

        self.hparams = {k:v for k, v in local_vars.items() \
            if k not in set(ignore+["self"]) and not k.startswith("_")}

        for k, v in self.hparams.items():
            setattr(self, k, v)

class DataModule(Hyperparameters):
    def __init__(self, root="../data", num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError
    
    def train_dataloader(self):
        return self.get_dataloader(train=True)
    
    def val_dataloader(self):
        return self.get_dataloader(train=False)
    
    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)

        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)