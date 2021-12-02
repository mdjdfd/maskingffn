import helper as hp
import numpy as np
import torch
import torch.utils
import deepstruct.sparse
from scipy.stats import rankdata
from collections import OrderedDict
from deepstruct.learning import run_evaluation


def network_prune():
    batch_size = 10
    train_loader, test_loader = hp.get_mnist_loaders(batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature, labels = iter(train_loader).next()

    input_shape = feature.shape[1:]
    output_size = int(labels.shape[-1])

    model = deepstruct.sparse.MaskedDeepFFN(input_shape, output_size, [300, 100])

    model.load_state_dict(torch.load('cache/initial_model.pt'))
    weights = model.state_dict()

    layers = list(model.state_dict())

    ranks = {}
    pruned_weights = []

    for l in layers[:9:3]:
        if 'weight' in l or 'bias' in l:
            data = weights[l]

            w = np.array(data)

            ranks[l] = (rankdata(np.abs(w), method='dense') - 1).astype(int).reshape(w.shape)

            lower_bound_rank = np.ceil(np.max(ranks[l]) * 0.50).astype(int)  # 50% Pruning

            ranks[l][ranks[l] <= lower_bound_rank] = 0
            ranks[l][ranks[l] > lower_bound_rank] = 1

            w = w * ranks[l]

            data[...] = torch.from_numpy(w)

            pruned_weights.append(data)

    print(pruned_weights)

    pruned_weights.append(weights[layers[-3]])  # Output layer weights

    new_state_dict = OrderedDict()

    for l, pw in zip(layers, pruned_weights):
        new_state_dict[l] = pw

    model.state_dict = new_state_dict

    assert run_evaluation(test_loader, model, device) > 1 / output_size
    test_accuracy = run_evaluation(test_loader, model, device)
    print(test_accuracy)
