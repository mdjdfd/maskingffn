import training as tr
import pruning as pr
import visualization as vis
import torch
import os
import helper as hp
import clustering as cl
import parameter as prm


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # prm.store_param()

    test_accuracy, model = tr.config_and_train()
    # pr.network_prune()

    # tr.config_and_train()

    # wt.model_init()

    # cl.model_reading()

    # Visualization calls
    # vis.model_visualization_tensorboard(model)                       #Visualization model training in tensorboard
    # vis.single_image_visualization()                                 #Image ploting with matlib
    # vis.grid_image_visualization()                                   #Image ploting with tensorboard
    # vis.grid_image_visualization_tensorboard()                       #Grid Visualization


if __name__ == "__main__":
    main()
