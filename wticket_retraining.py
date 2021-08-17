import helper as hp
import torch
import torch.utils
import torch.nn as nn
import deepstruct.sparse
import pickle
import json
import numpy as np
import os

from tqdm import tqdm
from deepstruct.learning import train
from deepstruct.learning import run_evaluation


def retrain_wticket():
    base_path = "/Users/junaidfahad/Downloads/Masters/Master Thesis Proposal/experiment_data/storage/2021-08-12-193331-823d8a89-bee8-41b5-9d2b-a15bf7daa077/"
    wticket_model_path = base_path + "initial_model.pt"
    wticket_mask_path = base_path + "9/lt_mask_2.0.pkl"

    batch_size = 60
    train_loader, test_loader = hp.get_mnist_loaders(batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature, labels = iter(train_loader).next()

    input_shape = feature.shape[1:]
    output_size = int(labels.shape[-1])

    loaded_model = deepstruct.sparse.MaskedDeepFFN(input_shape, output_size, [300, 100])
    loaded_model.load_state_dict(torch.load(wticket_model_path, map_location='cpu'))
    loaded_model.eval()

    with open(wticket_mask_path, 'rb') as f:
        mask = pickle.load(f)

    weights = loaded_model.state_dict()

    training_epochs = 30

    learning_rate = 0.01
    optimizer = torch.optim.SGD(loaded_model.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()

    original_initialization(loaded_model, mask, weights)

    train_loss_arr = np.zeros(training_epochs, float)
    test_accuracy_arr = np.zeros(training_epochs, float)
    progress_bar = tqdm(range(training_epochs))

    path_retraining = os.path.join(base_path, "retraining")
    if not os.path.exists(path_retraining):
        os.makedirs(path_retraining)

    path_retraining = os.path.join(path_retraining, "9")  # Change folder no
    if not os.path.exists(path_retraining):
        os.makedirs(path_retraining)

    for train_epoch in progress_bar:
        accuracy = run_evaluation(test_loader, loaded_model, device)
        test_accuracy_arr[train_epoch] = accuracy

        train_loss, train_accuracy = train(train_loader, loaded_model, optimizer, loss, device)
        train_loss_arr[train_epoch] = train_loss

        progress_bar.set_description(
            f'Train Epoch: {train_epoch + 1}/{training_epochs} Loss: {train_loss:.6f} Accuracy: {accuracy:.2f}%')

    store_retraining_data(train_loss_arr, test_accuracy_arr, path_retraining)

    # current_mask = mask
    # for training_iteration in range(0, ITERATION):
    #     current_mask = prune_by_percentile(loaded_model, current_mask, prune_percentile)
    #     retraining_loop(train_loader, test_loader, loaded_model, optimizer, loss, device,
    #                     weights,
    #                     mask,
    #                     training_epochs)


# def retraining_loop(train_loader, test_loader, loaded_model, optimizer, loss, device,
#                     weights,
#                     current_mask,
#                     training_epochs):
#     original_initialization(loaded_model, current_mask, weights)
#
#     progress_bar = tqdm(range(training_epochs))
#
#     for train_epoch in progress_bar:
#         accuracy = run_evaluation(test_loader, loaded_model, device)
#
#         train_loss, train_accuracy = train(train_loader, loaded_model, optimizer, loss, device)
#
#         progress_bar.set_description(
#             f'Train Epoch: {train_epoch + 1}/{training_epochs} Loss: {train_loss:.6f} Accuracy: {accuracy:.2f}%')
#
#


def store_retraining_data(train_loss_arr, test_accuracy_arr, path_retraining):
    param = {'train_loss_arr': train_loss_arr.tolist(), 'test_accuracy_arr': test_accuracy_arr.tolist()}

    with open(f"{path_retraining}/retraining_data_12.json", 'w') as fp:  # Change json file no
        json.dump(param, fp, sort_keys=True, indent=4)


def original_initialization(model, mask, weights):
    index = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask[index] * weights[name].cpu().numpy()).to(weight_dev)
            index = index + 1
        if 'bias' in name:
            param.data = weights[name]


#
#
# def prune_by_percentile(original_model, current_mask, percent, resample=False, reinit=False, **kwargs):
#     index = 0
#     for name, param in original_model.named_parameters():
#         if 'weight' in name:
#             tensor = param.data.cpu().numpy()
#             alive = tensor[np.nonzero(tensor)]
#             percentile_value = np.percentile(abs(alive), percent)
#
#             weight_dev = param.device
#             new_mask = np.where(abs(tensor) < percentile_value, 0, current_mask[index])
#
#             param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
#             current_mask[index] = new_mask
#             index = index + 1
#
#     return current_mask


def print_model(loaded_model):
    weights = loaded_model.state_dict()
    layers = list(loaded_model.state_dict())
    for l in layers[:9:3]:
        if 'weight' in l or 'bias' in l:
            data = weights[l]
            print(data)
