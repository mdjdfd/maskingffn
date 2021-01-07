import visualization as vis
import masking as rm


def main():
    accuracy, masked_tensor, model = rm.random_masking_first_layer()   #Training the model
    vis.model_visualization_tensorboard(model)                         #Visualization model training in tensorboard
    # vis.single_image_visualization()                                 #Image ploting with matlib
    # vis.grid_image_visualization()                                   #Image ploting with tensorboard
    # vis.grid_image_visualization_tensorboard()                       #Grid Visualization


if __name__ == "__main__":
    main()
