import torch
import torch.utils
import torch.nn as nn
import deepstruct.sparse

from deepstruct.learning import train
import helper as hp
from deepstruct.learning import run_evaluation


def config_and_train():
    batch_size = 10
    train_loader, test_loader = hp.get_mnist_loaders(batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature, labels = iter(train_loader).next()

    input_shape = feature.shape[1:]
    output_size = int(labels.shape[-1])

    loss = torch.nn.CrossEntropyLoss()

    epochs = 4
    model = deepstruct.sparse.MaskedDeepFFN(input_shape, output_size, [300, 100])
    model.to(device)
    model.apply(weight_init)

    torch.save(model.state_dict(), 'cache/initial_model.pt')

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        print(train(train_loader, model, optimizer, loss, device))
        torch.save(model.state_dict(), 'cache/model.pt')

    assert run_evaluation(test_loader, model, device) > 1 / output_size
    test_accuracy = run_evaluation(test_loader, model, device)

    model.apply_mask()
    model.recompute_mask(theta=0.01)

    return test_accuracy, model

    # n = 100
    # d = 784
    # random_mat = torch.rand(n, d)
    # k = round(0.50 * d)
    # k_th_quantile = torch.topk(random_mat, k, largest=False)[0][:, -1:]
    # bool_tensor = random_mat <= k_th_quantile
    # random_mask = torch.where(bool_tensor, torch.tensor(1), torch.tensor(0))
    #
    # first_layer_mask = model.layer_first.get_mask()
    #
    # result_mask = random_mask * first_layer_mask


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0, 0.3)
        nn.init.normal_(m.bias.data, 0, 0.3)
