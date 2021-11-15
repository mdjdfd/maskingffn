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


def retrain_wticket(prune_iteration, training_iteration):
    base_path = "/Volumes/Extreme SSD/thesis_storage/storage/2021-10-03-141255-d6177e91-2c06-430b-9ea2-1c05a78839ed/"
    wticket_model_path = base_path + "initial_model.pt"
    wticket_mask_path = base_path + str(prune_iteration) + "/lt_mask_73.0.pkl"

    batch_size = 100
    train_loader, test_loader = hp.get_mnist_loaders(batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature, labels = iter(train_loader).next()

    input_shape = feature.shape[1:]
    output_size = int(labels.shape[-1])

    # Load stored model
    # hidden_layer = [300, 200, 100, 50]
    # loaded_model = deepstruct.sparse.MaskedDeepFFN(input_shape, output_size, hidden_layer)
    # loaded_model.load_state_dict(torch.load(wticket_model_path, map_location='cpu'))
    # loaded_model.eval()

    # Random re-initialization
    hidden_layer = [300, 200, 100, 50]
    loaded_model = deepstruct.sparse.MaskedDeepFFN(input_shape, output_size, hidden_layer)
    loaded_model.to(device)

    # mask = create_random_mask(loaded_model)

    with open(wticket_mask_path, 'rb') as f:
        mask = pickle.load(f)

    weights = loaded_model.state_dict()

    training_epochs = 50

    learning_rate = 1.2e-3
    optimizer = torch.optim.Adam(loaded_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss = nn.CrossEntropyLoss()

    original_initialization(loaded_model, mask, weights)

    train_loss_arr = np.zeros(training_epochs, float)
    test_accuracy_arr = np.zeros(training_epochs, float)
    progress_bar = tqdm(range(training_epochs))

    # path_retraining = os.path.join(base_path, "retraining")
    path_retraining = os.path.join(base_path, "retraining/random_model")
    # path_retraining = os.path.join(base_path, "retraining/random_mask")

    if not os.path.exists(path_retraining):
        os.makedirs(path_retraining)

    path_retraining = os.path.join(path_retraining, str(prune_iteration))
    if not os.path.exists(path_retraining):
        os.makedirs(path_retraining)

    for train_epoch in progress_bar:
        accuracy = run_evaluation(test_loader, loaded_model, device)
        test_accuracy_arr[train_epoch] = accuracy

        train_loss = train(loaded_model, train_loader, optimizer, loss, device)
        train_loss_arr[train_epoch] = train_loss

        progress_bar.set_description(
            f'Train Epoch: {train_epoch + 1}/{training_epochs} Loss: {train_loss:.6f} Accuracy: {accuracy:.2f}%')

    store_retraining_data(train_loss_arr, test_accuracy_arr, path_retraining, training_iteration)


# Function for Training
def train(model, train_loader, optimizer, criterion, device):
    EPS = 1e-6
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


def store_retraining_data(train_loss_arr, test_accuracy_arr, path_retraining, training_iteration):
    param = {'train_loss_arr': train_loss_arr.tolist(), 'test_accuracy_arr': test_accuracy_arr.tolist()}

    with open(f"{path_retraining}/retraining_data_{training_iteration}.json", 'w') as fp:  # Change json file no
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


def print_model(loaded_model):
    weights = loaded_model.state_dict()
    layers = list(loaded_model.state_dict())
    for l in layers[:9:3]:
        if 'weight' in l or 'bias' in l:
            data = weights[l]
            print(data)


def create_random_mask(model):
    index = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            index = index + 1

    random_mask = [None] * index

    index = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            random_mask[index] = np.random.choice([0, 1], tensor.shape).astype('f')
            index = index + 1

    return random_mask
