import torch
import torch.utils
import deepstruct.sparse
import helper as hp
import torch.nn as nn
import numpy as np
import copy
from tqdm import tqdm
import os
import utils
import pickle
import json
from deepstruct.learning import train
from deepstruct.learning import run_evaluation
import matplotlib.pyplot as plt


def run_model(storage_path):
    batch_size = 10

    train_loader, test_loader = hp.get_mnist_loaders(batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feature, labels = iter(train_loader).next()
    assert len(feature) == len(labels)

    input_shape = feature.shape[1:]
    output_size = int(labels.shape[-1])

    # Network initialization
    hidden_layer = [300, 100]
    original_model = deepstruct.sparse.MaskedDeepFFN(input_shape, output_size, hidden_layer)
    assert isinstance(original_model, deepstruct.sparse.MaskedDeepFFN)

    # Load model into device
    original_model.to(device)

    # Normal distribution of weights
    original_model.apply(weight_init)

    # Storing initial network
    initial_state_dict = copy.deepcopy(original_model.state_dict())
    torch.save(initial_state_dict, f'{storage_path}/initial_model.pt')

    prune_type = "lt"
    # prune_type = "reinit"

    # Same sized empty mask creation
    initial_mask = create_mask(original_model)

    # Stochastic Gradient Descent optimization and cross entropy loss
    learning_rate = 0.01
    optimizer = torch.optim.SGD(original_model.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()

    # Close all open figures and set to matplotlib
    plt.switch_backend('agg')

    # Start of Pruning Functionality
    # best_accuracy = 0
    prune_percentile = 10
    ITERATION = 10
    training_epochs = 4

    # Store hyperparameter
    store_hyperparameter(batch_size, hidden_layer, learning_rate, prune_type, prune_percentile, ITERATION,
                         training_epochs, storage_path)

    # Initial training
    initial_training(train_loader, test_loader, original_model, optimizer, loss, device,
                     training_epochs, storage_path)

    # Iterative pruning
    current_mask = initial_mask
    for training_iteration in range(0, ITERATION):
        current_mask = prune_by_percentile(original_model, current_mask, prune_percentile)
        winning_ticket_loop(train_loader, test_loader, original_model, optimizer, loss, device,
                            initial_state_dict,
                            current_mask,
                            training_epochs,
                            prune_type, training_iteration, storage_path)


def initial_training(train_loader, test_loader, original_model, optimizer, loss, device, training_epochs, storage_path):
    utils.print_nonzeros(original_model)
    progress_bar = tqdm(range(training_epochs))
    for train_epoch in progress_bar:
        train_loss, train_accuracy = train(train_loader, original_model, optimizer, loss, device)
        accuracy = run_evaluation(test_loader, original_model, device)
        torch.save(original_model.state_dict(), f'{storage_path}/initial_trained_model.pt')
        progress_bar.set_description(
            f'Train Epoch: {train_epoch + 1}/{training_epochs} Loss: {train_loss:.6f} Accuracy: {accuracy:.2f}%')


def winning_ticket_loop(train_loader, test_loader, original_model, optimizer, loss, device,
                        initial_weights,
                        current_mask,
                        training_epochs,
                        prune_type, training_iteration, storage_path):
    train_loss_arr = np.zeros(training_epochs, float)
    test_accuracy_arr = np.zeros(training_epochs, float)

    original_initialization(original_model, current_mask, initial_weights)

    pruned_mask = utils.print_nonzeros(original_model)
    progress_bar = tqdm(range(training_epochs))

    path_experiment = os.path.join(storage_path, str(training_iteration))
    os.makedirs(path_experiment)

    for train_epoch in progress_bar:
        train_loss, train_accuracy = train(train_loader, original_model, optimizer, loss, device)
        train_loss_arr[train_epoch] = train_loss

        accuracy = run_evaluation(test_loader, original_model, device)
        test_accuracy_arr[train_epoch] = accuracy

        torch.save(original_model.state_dict(),
                   f'{path_experiment}/{prune_type}_train_epoch_{train_epoch}.pt')

        progress_bar.set_description(
            f'Train Epoch: {train_epoch + 1}/{training_epochs} Loss: {train_loss:.6f} Accuracy: {accuracy:.2f}%')

    store_training_data(train_loss_arr, test_accuracy_arr, path_experiment)

    drawing_plots(training_epochs, train_loss_arr, test_accuracy_arr, pruned_mask, path_experiment)

    with open(f"{os.getcwd()}/{path_experiment}/{prune_type}_mask_{pruned_mask}.pkl", 'wb') as fp:
        pickle.dump(pruned_mask, fp)


def drawing_plots(training_epochs, train_loss_arr, test_accuracy_arr, pruned_mask, path_experiment):
    plt.plot(np.arange(1, training_epochs + 1),
             100 * (train_loss_arr - np.min(train_loss_arr)) / np.ptp(train_loss_arr).astype(float), c="blue",
             label="Loss")
    plt.plot(np.arange(1, training_epochs + 1), test_accuracy_arr, c="red", label="Accuracy")
    plt.title(f"Loss Vs Accuracy Vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss and Accuracy")
    plt.legend()
    plt.grid(color="gray")
    plt.savefig(f"{os.getcwd()}/{path_experiment}/LossVsAccuracy_{pruned_mask}.png", dpi=1200)
    plt.close()


def store_training_data(train_loss_arr, test_accuracy_arr, path_experiment):
    param = {'train_loss_arr': train_loss_arr.tolist(), 'test_accuracy_arr': test_accuracy_arr.tolist()}

    with open(f"{os.getcwd()}/{path_experiment}/training_data.json", 'w') as fp:
        json.dump(param, fp, sort_keys=True, indent=4)


def store_hyperparameter(batch_size, hidden_layer, learning_rate, prune_type, prune_percentile, ITERATION,
                         training_epochs, storage_path):
    param = {'batch_size': batch_size, 'hidden_layer': hidden_layer, 'learning_rate': learning_rate,
             'prune_type': prune_type, 'prune_percentile': prune_percentile, 'ITERATION': ITERATION,
             'training_epochs': training_epochs}

    with open(f"{os.getcwd()}/{storage_path}/parameters.json", 'w') as fp:
        json.dump(param, fp, sort_keys=True, indent=4)


def prune_by_percentile(original_model, current_mask, percent, resample=False, reinit=False, **kwargs):
    index = 0
    for name, param in original_model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]
            percentile_value = np.percentile(abs(alive), percent)

            weight_dev = param.device
            new_mask = np.where(abs(tensor) < percentile_value, 0, current_mask[index])

            param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            current_mask[index] = new_mask
            index = index + 1

    return current_mask


def original_initialization(original_model, mask_temp, initial_state_dict):
    index = 0
    for name, param in original_model.named_parameters():
        if 'weight' in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[index] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            index = index + 1
        if 'bias' in name:
            param.data = initial_state_dict[name]


def create_mask(original_model):
    index = 0
    for name, param in original_model.named_parameters():
        if 'weight' in name:
            index = index + 1

    local_mask = [None] * index

    index = 0
    for name, param in original_model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            local_mask[index] = np.ones_like(tensor)
            index = index + 1

    return local_mask


def weight_init(m):
    if isinstance(m, nn.Linear):
        assert isinstance(m, nn.Linear)
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
