import torch
import torch.utils
import deepstruct.sparse
from deepstruct.learning import train
import helper as hp
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def model_init():
    batch_size = 10
    train_loader, test_loader = hp.get_mnist_loaders(batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature, labels = iter(train_loader).next()
    assert len(feature) == len(labels)

    input_shape = feature.shape[1:]
    output_size = int(labels.shape[-1])

    model = deepstruct.sparse.MaskedDeepFFN(input_shape, output_size, [100, 50, 10])
    assert isinstance(model, deepstruct.sparse.MaskedDeepFFN)

    model.to(device)

    model.apply(weight_init)



def weight_init(m):
    if isinstance(m, nn.Linear):
        assert isinstance(m, nn.Linear)
        nn.init.normal_(m.weight.data, 0, 0.3)
