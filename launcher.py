import training as tr
import pruning as pr
import visualization as vis
import torch
import os
import helper as hp
import wticket_retraining as ret
import parameter as prm
import pickle
import pruning as pr

import graph_visualization as gv

def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    prm.store_param()

    # for i in range(10):
    #     ret.retrain_wticket(2, i+1)

    # gv.show_network_graph()

    # test_accuracy, model = tr.config_and_train()
    # pr.network_prune()

    # tr.config_and_train()
    # pr.network_prune()

    # wt.model_init()

    # cl.model_reading()

    # Visualization calls
    # vis.model_visualization_tensorboard(model)                       #Visualization model training in tensorboard
    # vis.single_image_visualization()                                 #Image ploting with matlib
    # vis.grid_image_visualization()                                   #Image ploting with tensorboard
    # vis.grid_image_visualization_tensorboard()                       #Grid Visualization


if __name__ == "__main__":
    main()
