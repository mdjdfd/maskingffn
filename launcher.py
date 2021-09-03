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


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    prm.store_param()
    # ret.retrain_wticket()


    # wticket_mask_path = "storage/2021-07-11-111745-fe392342-d71c-4ee8-a1fe-712f13e9a884/0/lt_mask_90.0.pkl"
    # mask = []
    # with open(wticket_mask_path, 'rb') as f:
    #     mask.append(pickle.load(f))
    # wticket_retraining.Retrain(mask)

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
