import helper as hp
import matplotlib.pyplot as plt


def data_image_visualization():
    batch_size = 10
    train_loader, test_loader = hp.get_mnist_loaders(batch_size)
    feature, labels = iter(train_loader).next()
    plt.imshow(feature[9][0])
    print(labels[9])
    plt.show()
