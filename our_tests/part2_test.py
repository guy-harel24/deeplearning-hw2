import os
import torch
import unittest
import hw2.optimizers as optimizers
import hw2.layers as layers
import hw2.answers as answers
import hw2.training as training
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as tvtf

data_dir = os.path.expanduser('~/.pytorch-datasets')
ds_train = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=True, transform=tvtf.ToTensor())
ds_test = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=False, transform=tvtf.ToTensor())
seed = 42
plt.rcParams.update({'font.size': 12})
test = unittest.TestCase()

class TestOptimizers(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)


    def test_vanilla_sgd(self):
        # Test VanillaSGD
        torch.manual_seed(42)
        p = torch.randn(500, 10)
        dp = torch.randn(*p.shape)*2
        params = [(p, dp)]
        vsgd = optimizers.VanillaSGD(params, learn_rate=0.5, reg=0.1)
        vsgd.step()

        expected_p = torch.load('tests/assets/expected_vsgd.pt')
        diff = torch.norm(p-expected_p).item()
        print(f'diff={diff}')
        assert(diff < 1e-3)

    def test_layer_trainer(self):
        # Overfit to a very small dataset of 20 samples
        batch_size = 10
        max_batches = 2
        dl_train = DataLoader(ds_train, batch_size, shuffle=False)

        # Get hyperparameters
        hp = answers.part2_overfit_hp()

        torch.manual_seed(seed)

        # Build a model and loss using our custom MLP and CE implementations
        model = layers.MLP(3*32*32, num_classes=10, hidden_features=[128]*3, wstd=hp['wstd'])
        loss_fn = layers.CrossEntropyLoss()

        # Use our custom optimizer
        optimizer = optimizers.VanillaSGD(model.params(), learn_rate=hp['lr'], reg=hp['reg'])

        # Run training over small dataset multiple times
        trainer = training.LayerTrainer(model, loss_fn, optimizer)
        best_acc = 0
        for i in range(20):
            res = trainer.train_epoch(dl_train, max_batches=max_batches)
            best_acc = res.accuracy if res.accuracy > best_acc else best_acc
            
        assert best_acc >= 98