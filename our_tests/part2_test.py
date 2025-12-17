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
batch_size = 50
max_batches = 100
in_features = 3*32*32
num_classes = 10
dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size//2, shuffle=False)

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
    
    def test_dropout_layer(self):
        from hw2.grad_compare import compare_layer_to_torch
        # Check architecture of MLP with dropout layers
        mlp_dropout = layers.MLP(in_features, num_classes, [50]*3, dropout=0.6)
        print(mlp_dropout)
        test.assertEqual(len(mlp_dropout.sequence), 10)
        for b1, b2 in zip(mlp_dropout.sequence, mlp_dropout.sequence[1:]):
            if str(b1).lower() == 'relu':
                test.assertTrue(str(b2).startswith('Dropout'))
        test.assertTrue(str(mlp_dropout.sequence[-1]).startswith('Linear'))
        # Test end-to-end gradient in train and test modes.
        print('Dropout, train mode')
        mlp_dropout.train(True)
        for diff in compare_layer_to_torch(mlp_dropout, torch.randn(500, in_features)):
            test.assertLess(diff, 1e-3)
            
        print('Dropout, test mode')
        mlp_dropout.train(False)
        for diff in compare_layer_to_torch(mlp_dropout, torch.randn(500, in_features)):
            test.assertLess(diff, 1e-3)