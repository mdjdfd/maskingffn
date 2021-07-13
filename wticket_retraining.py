import helper as hp
import torch
import torch.utils
import torch.nn as nn
import deepstruct.sparse
import pickle
import utils
import numpy as np
from tqdm import tqdm
from deepstruct.learning import train
from deepstruct.learning import run_evaluation



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

    mask = []
    with open(wticket_mask_path, 'rb') as f:
        mask.append(pickle.load(f))

    model.apply_mask(mask)
    # weights = model.state_dict()
    #
    # layers = list(model.state_dict())
    #
    # for l in layers[:9:3]:
    #     if 'weight' in l:
    #         print(weights[l])

    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()

    training_epochs = 50
    do_retraining(train_loader, test_loader, model, optimizer, loss, device, training_epochs)


def do_retraining(train_loader, test_loader, wticket_model, optimizer, loss, device, training_epochs):
    train_loss_arr = np.zeros(training_epochs, float)
    test_accuracy_arr = np.zeros(training_epochs, float)

    utils.print_nonzeros(wticket_model)
    progress_bar = tqdm(range(training_epochs))

    for train_epoch in progress_bar:
        accuracy = run_evaluation(test_loader, wticket_model, device)
        test_accuracy_arr[train_epoch] = accuracy

        train_loss, train_accuracy = train(train_loader, wticket_model, optimizer, loss, device)
        train_loss_arr[train_epoch] = train_loss

        progress_bar.set_description(
            f'Train Epoch: {train_epoch + 1}/{training_epochs} Loss: {train_loss:.6f} Accuracy: {accuracy:.2f}%')