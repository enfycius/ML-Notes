import torch
import inspect
import collections

import numpy as np

from torch import nn

from IPython import display
from matplotlib import pyplot as plt


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
    
class ProgressBoard(Hyperparameters):
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale="linear", yscale="linear",
                 ls=["-", "--", "-.", ":"], colors=["C0", "C1", "C2", "C3"],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, X, y, label, every_n=1):
        Point = collections.namedtuple("Point", ["x", "y"])

        if not hasattr(self, "raw_points"):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()

        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []

        points = self.raw_points[label]
        line = self.data[label]

        points.append(Point(X, y))

        if len(points) != every_n:
            return
        
        mean = lambda x: sum(x) / len(x)

        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points])))
        
        points.clear()

        if not self.display:
            return

        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)

        plt_lines, labels = [], []

        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v],
                                          linestyle=ls, color=color)[0])
            labels.append(k)

        axes = self.axes if self.axes else plt.gca()

        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x

        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        display.display(self.fig)
        display.clear_output(wait=True)

    
class Module(nn.Module, Hyperparameters):
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError
    
    def forward(self, X):
        assert hasattr(self, "net"), "Neural network is defined"
        return self.net(X)
    
    def plot(self, key, value, train):
        assert hasattr(self, "trainer"), "Trainer is not inited"
        self.board.xlabel = "epoch"

        if train:
            X  = self.trainer.train_batch_idx / \
            self.trainer.num_train_batches

            n = self.trainer.num_train_batches / \
            self.plot_train_per_epoch

        else:
            X = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
            self.plot_valid_per_epoch

        value_np = value.detach().cpu().numpy() if value.is_cuda else value.detach().cpu().numpy()
        self.board.draw(X, value_np, ("train_" if train else "val_") + key, \
                        every_n=int(n))
        
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot("loss", l, train=True)

        return l
    
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot("loss", l, train=False)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
    
    def apply_init(self, inputs, init=None):
        self.forward(*inputs)

        if init is not None:
            self.net.apply(init)

class Trainer(Hyperparameters):
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, "No GPU support yet"


    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader) \
                                if self.val_dataloader is not None else 0)
        
    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0

        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
    
    def prepare_batch(self, batch):
        return batch
    
    def fit_epoch(self):
        self.model.train()

        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()

            with torch.no_grad():
                loss.backward()

                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)

                self.optim.step()

            self.train_batch_idx += 1

        if self.val_dataloader is None:
            return
        
        self.model.eval()

        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))

            self.val_batch_idx += 1

    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()

        self.gpus = [torch.device(f"cuda:{i}") for i in range(min(num_gpus, torch.cuda.device_count()))] \
                    if num_gpus > 0 else [torch.device("cpu")]
        
    def prepare_batch(self, batch):
        if self.gpus:
            batch = [a.to(self.gpus[0]) for a in batch]

        return batch
    
    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]

        if self.gpus:
            model.to(self.gpus[0])

        self.model = model

    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))

        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm
