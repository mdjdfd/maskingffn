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
import torch.nn.functional as F


################### Algorithm Steps ####################
# 1. Random network initialization
# 2. Initial training of the network with specific number of epochs
# 3. Pruning a percentage of network which gives us an updated mask
# 4. Reset the survived network to original initialization before training again


def run_model(storage_path):
    batch_size = 10

    train_loader, test_loader = hp.get_mnist_loaders(batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feature, labels = iter(train_loader).next()
    assert len(feature) == len(labels)

    input_shape = feature.shape[1:]
    output_size = int(labels.shape[-1])

    # Network initialization
    # hidden_layer = [300, 100]
    hidden_layer = [300, 200, 100, 50]
    original_model = deepstruct.sparse.MaskedDeepFFN(input_shape, output_size, hidden_layer)
    assert isinstance(original_model, deepstruct.sparse.MaskedDeepFFN)

    # Load model into device
    original_model.to(device)

    # Normal distribution of weights
    original_model.apply(weight_init)


    # Storing initial network
    initial_state_dict = copy.deepcopy(original_model.state_dict())
    model = original_model.state_dict()
    torch.save(model, f'{storage_path}/initial_model.pt')

    prune_type = "lt"
    # prune_type = "reinit"

    # Same sized empty mask creation
    initial_mask = create_mask(original_model)

    # Stochastic Gradient Descent optimization and cross entropy loss
    learning_rate = 0.01
    optimizer = torch.optim.Adam(original_model.parameters(), weight_decay=1e-4)
    loss = nn.CrossEntropyLoss()

    # Close all open figures and set to matplotlib
    plt.switch_backend('agg')

    # Start of Pruning Functionality
    # best_accuracy = 0
    prune_percentile = 10
    ITERATION = 10
    training_epochs = 50

    # Store hyperparameter
    store_hyperparameter(batch_size, hidden_layer, learning_rate, prune_type, prune_percentile, ITERATION,
                         training_epochs, storage_path)


    # Initial training
    initial_training(original_model, train_loader, test_loader, optimizer, loss, device,
                     training_epochs, prune_type, storage_path)



    # Iterative pruning
    current_mask = initial_mask
    for training_iteration in range(0, ITERATION):
        current_mask = prune_by_percentile(original_model, current_mask, prune_percentile)
        winning_ticket_loop(original_model, train_loader, test_loader, optimizer, loss, device,
                            initial_state_dict,
                            training_epochs,
                            prune_type, training_iteration, current_mask, storage_path)


def initial_training(original_model, train_loader, test_loader, optimizer, loss, device,
                     training_epochs, prune_type, storage_path):
    train_loss_arr = np.zeros(training_epochs, float)
    test_accuracy_arr = np.zeros(training_epochs, float)

    utils.print_nonzeros(original_model)
    progress_bar = tqdm(range(training_epochs))

    path_experiment = os.path.join(storage_path, "initial_training")
    os.makedirs(path_experiment)

    for train_epoch in progress_bar:
        accuracy = run_evaluation(test_loader, original_model, device)
        # accuracy = test(original_model, test_loader, loss)
        test_accuracy_arr[train_epoch] = accuracy

        # train_loss, train_accuracy = train(train_loader, original_model, optimizer, loss, device)
        train_loss = train(original_model, train_loader, optimizer, loss)
        train_loss_arr[train_epoch] = train_loss

        torch.save(original_model.state_dict(),
                   f'{path_experiment}/{prune_type}_train_epoch_{train_epoch}.pt')

        progress_bar.set_description(
            f'Train Epoch: {train_epoch + 1}/{training_epochs} Loss: {train_loss:.6f} Accuracy: {accuracy:.2f}%')

    store_training_data(train_loss_arr, test_accuracy_arr, path_experiment)


def winning_ticket_loop(original_model, train_loader, test_loader, optimizer, loss, device,
                        initial_weights,
                        training_epochs,
                        prune_type, training_iteration, mask, storage_path):
    train_loss_arr = np.zeros(training_epochs, float)
    test_accuracy_arr = np.zeros(training_epochs, float)

    original_initialization(original_model, mask, initial_weights)

    optimizer = torch.optim.Adam(original_model.parameters(), lr=1.2e-3, weight_decay=1e-4)

    pruned_mask = utils.print_nonzeros(original_model)
    progress_bar = tqdm(range(training_epochs))

    path_experiment = os.path.join(storage_path, str(training_iteration))
    os.makedirs(path_experiment)

    for train_epoch in progress_bar:
        accuracy = run_evaluation(test_loader, original_model, device)
        # accuracy = test(original_model, test_loader, loss)
        test_accuracy_arr[train_epoch] = accuracy

        # train_loss, train_accuracy = train(train_loader, original_model, optimizer, loss, device)
        train_loss = train(original_model, train_loader, optimizer, loss)
        train_loss_arr[train_epoch] = train_loss

        torch.save(original_model.state_dict(),
                   f'{path_experiment}/{prune_type}_train_epoch_{train_epoch}.pt')

        progress_bar.set_description(
            f'Train Epoch: {train_epoch + 1}/{training_epochs} Loss: {train_loss:.6f} Accuracy: {accuracy:.2f}%')

    store_training_data(train_loss_arr, test_accuracy_arr, path_experiment)

    drawing_plots(training_epochs, train_loss_arr, test_accuracy_arr, pruned_mask, path_experiment)

    with open(f"{os.getcwd()}/{path_experiment}/{prune_type}_mask_{pruned_mask}.pkl", 'wb') as fp:
        pickle.dump(mask, fp)






# Function for Training
def train(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        # imgs, targets = next(train_loader)
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()
    return train_loss.item()






##############Debug Purpose Only#############
def print_model(original_model):
    weights = original_model.state_dict()
    layers = list(original_model.state_dict())
    for l in layers[:]:
        if 'weight' in l:
            data = weights[l]
            print(data)


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


def prune_by_percentile(original_model, mask, percent, resample=False, reinit=False, **kwargs):
    step = 0
    for name, param in original_model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]
            percentile_value = np.percentile(abs(alive), percent)

            weight_dev = param.device
            new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

            param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            mask[step] = new_mask
            step += 1
    return mask


def original_initialization(original_model, mask_temp, initial_state_dict):
    step = 0
    for name, param in original_model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0


def create_mask(original_model):
    index = 0
    for name, param in original_model.named_parameters():
        if 'weight' in name:
            index = index + 1

    mask = [None] * index

    index = 0
    for name, param in original_model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[index] = np.ones_like(tensor)
            index = index + 1

    return mask


def weight_init(m):
    if isinstance(m, nn.Linear):
        assert isinstance(m, nn.Linear)
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
