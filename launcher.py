import visualization as vis
import masking as rm


def main():
    accuracy, masked_tensor, model = rm.random_masking_first_layer()
    vis.model_visualization_tensorboard(model)
    # vis.single_image_visualization()
    # vis.grid_image_visualization()
    # vis.grid_image_visualization_tensorboard()


if __name__ == "__main__":
    main()
