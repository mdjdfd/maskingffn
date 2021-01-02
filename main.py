import torch
import torch.utils

import deepstruct.sparse

from deepstruct.learning import train
from helper import get_mnist_loaders


def random_masking_first_layer():
    batch_size = 10
    train_loader, test_loader = get_mnist_loaders(batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature, labels = iter(train_loader).next()

    input_shape = feature.shape[1:]
    output_size = int(labels.shape[-1])

    loss = torch.nn.CrossEntropyLoss()

    epochs = 3
    model = deepstruct.sparse.MaskedDeepFFN(input_shape, output_size, [100, 50, 10])
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        train(train_loader, model, optimizer, loss, device)

    model.apply_mask()
    model.recompute_mask(theta=0.01)

    n = 100
    d = 784
    random_mat = torch.rand(n, d)
    k = round(0.50 * d)
    k_th_quantile = torch.topk(random_mat, k, largest=False)[0][:, -1:]
    bool_tensor = random_mat <= k_th_quantile
    random_mask = torch.where(bool_tensor, torch.tensor(1), torch.tensor(0))

    first_layer_mask = model.layer_first.get_mask()

    result_mask = random_mask * first_layer_mask

    return result_mask


print(random_masking_first_layer())
