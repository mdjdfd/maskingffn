import training as tr
import pruning as pr
import visualization as vis


def main():
    # test_accuracy, model = tr.config_and_train()

    pr.network_prune()

    # Visualization calls
    # vis.model_visualization_tensorboard(model)                       #Visualization model training in tensorboard
    # vis.single_image_visualization()                                 #Image ploting with matlib
    # vis.grid_image_visualization()                                   #Image ploting with tensorboard
    # vis.grid_image_visualization_tensorboard()                       #Grid Visualization


if __name__ == "__main__":
    main()
