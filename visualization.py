import helper as hp
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np


def mnist_dataset():
    batch_size = 10
    train_loader, test_loader = hp.get_mnist_loaders(batch_size)
    features, labels = iter(train_loader).next()

    return features, labels


def single_image_visualization():
    features, labels = mnist_dataset()
    plt.imshow(features[9][0])
    print(labels[9])
    plt.show()


def grid_image_visualization():
    features, labels = mnist_dataset()
    img_grid = torchvision.utils.make_grid(features)
    matplotlib_imshow(img_grid, one_channel=True)
    plt.show()


# Command to launch tensorboard localhost: tensorboard --logdir=runs
def grid_image_visualization_tensorboard():
    writer = SummaryWriter('runs/digits_mnist_visualization')

    features, labels = mnist_dataset()
    img_grid = torchvision.utils.make_grid(features)
    writer.add_image('digits', img_grid)
    writer.flush()


def model_visualization_tensorboard(model):
    writer = SummaryWriter('runs/digits_mnist_visualization')

    features, labels = mnist_dataset()
    writer.add_graph(model, features)
    writer.close()


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
