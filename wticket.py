import torch
import torch.utils
import deepstruct.sparse
import helper as hp
import torch.nn as nn
import numpy as np
import copy
from tqdm import tqdm
import utils
from deepstruct.learning import train
from deepstruct.learning import run_evaluation


def model_init():
    batch_size = 10

    train_loader, test_loader = hp.get_mnist_loaders(batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature, labels = iter(train_loader).next()
    assert len(feature) == len(labels)

    input_shape = feature.shape[1:]
    output_size = int(labels.shape[-1])


    #Network initialization
    global model
    model = deepstruct.sparse.MaskedDeepFFN(input_shape, output_size, [300, 100])
    assert isinstance(model, deepstruct.sparse.MaskedDeepFFN)

    #Normal distribution of weights
    model.apply(weight_init)


    #Storing initial network
    initial_state_dict = copy.deepcopy(model.state_dict())
    torch.save(initial_state_dict, 'cache/model.pt')


    #Same sized empty mask creation
    create_mask(model)

    #Stochastic Gradient Descent optimization and cross entropy loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()


    # Start of Pruning Functionality
    best_accuracy = 0
    ITERATION = 10
    comp = np.zeros(ITERATION, float)
    bestacc = np.zeros(ITERATION, float)
    epochs = 4
    all_loss = np.zeros(epochs, float)
    all_accuracy = np.zeros(epochs, float)

    for i in range(0, ITERATION):
        if not i == 0:              #Prune after initial training
            prune_by_percentile(10)
            original_initialization(mask, initial_state_dict)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        comp1 = utils.print_nonzeros(model)
        comp[i] = comp1
        pbar = tqdm(range(epochs))

        for j in pbar:

            accuracy = run_evaluation(test_loader, model, device)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), f'cache/{i}_model.pt')

            train_loss, train_accuracy = train(train_loader, model, optimizer, loss, device)
            all_loss[j] = train_loss
            all_accuracy[j] = accuracy

            pbar.set_description(
                f'Train Epoch: {j + 1}/{epochs} Loss: {train_loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')
        bestacc[i] = best_accuracy

        best_accuracy = 0
        all_accuracy = np.zeros(epochs, float)



def weight_init(m):
    if isinstance(m, nn.Linear):
        assert isinstance(m, nn.Linear)
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)


def original_initialization(mask_temp, initial_state_dict):
    global model

    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if 'bias' in name:
            param.data = initial_state_dict[name]
    step = 0


def create_mask(model):
    global step
    global mask
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            step = step + 1
    mask = [None] * step
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0


def prune_by_percentile(percent, resample=False, reinit=False, **kwargs):
    global step
    global mask
    global model

    # Calculate percentile value
    step = 0
    for name, param in model.named_parameters():

        # We do not prune bias term
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
            percentile_value = np.percentile(abs(alive), percent)

            # Convert Tensors to numpy and calculate
            weight_dev = param.device
            new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

            # Apply new weight and mask
            param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            mask[step] = new_mask
            step += 1
    step = 0