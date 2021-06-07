import helper as hp
import torch
import torch.utils
import deepstruct.sparse
import visualization as vis


def model_reading():

    batch_size = 10
    train_loader, test_loader = hp.get_mnist_loaders(batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature, labels = iter(train_loader).next()

    input_shape = feature.shape[1:]
    output_size = int(labels.shape[-1])
    model = deepstruct.sparse.MaskedDeepFFN(input_shape, output_size, [300, 100])

    # model.load_state_dict(torch.load('storage/2021-06-07-114257-e3d23a2a-ea5c-4358-9601-5cb443438810/initial_model.pt'))
    vis.model_visualization_tensorboard(model)
    # weights = model.state_dict()
    #
    # layers = list(model.state_dict())
    #
    # for l in layers[:9:3]:
    #     if 'weight' in l:
    #         print(weights[l])
