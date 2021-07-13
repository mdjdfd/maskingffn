import helper as hp
import torch
import torch.utils
import deepstruct.sparse


def retrain_wticket():
    wticket_model_path = "storage/2021-07-11-111745-fe392342-d71c-4ee8-a1fe-712f13e9a884/0/lt_train_epoch_37.pt"
    wticket_mask_path = "storage/2021-07-11-111745-fe392342-d71c-4ee8-a1fe-712f13e9a884/0/lt_mask_90.0.pkl"

    batch_size = 10
    train_loader, test_loader = hp.get_mnist_loaders(batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature, labels = iter(train_loader).next()

    input_shape = feature.shape[1:]
    output_size = int(labels.shape[-1])
    model = deepstruct.sparse.MaskedDeepFFN(input_shape, output_size, [300, 100])

    model.load_state_dict(torch.load(wticket_model_path))

    mask = model.load_state_dict(torch.load(wticket_mask_path))

    print(mask)


    # weights = model.state_dict()
    #
    # layers = list(model.state_dict())
    #
    # for l in layers[:9:3]:
    #     if 'weight' in l:
    #         print(weights[l])

