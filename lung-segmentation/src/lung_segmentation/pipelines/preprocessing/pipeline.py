from kedro.pipeline import Pipeline, node, pipeline
       
from .nodes import resize_images
from .nodes import convert_to_grayscale
from .nodes import normalize_images
from .nodes import group_to_numpy
from .nodes import feature_target_split
from .nodes import train_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=resize_images,
                inputs="lung_images",
                outputs="resized_lung_images",
                name="resize_lung_images_node",
                tags=["preprocessing"],
            ),
            node(
                func=convert_to_grayscale,
                inputs="resized_lung_images",
                outputs="grayscale_lung_images",
                name="grayscale_lung_images_node",
                tags=["preprocessing"],
            ),
            node(
                func=normalize_images,
                inputs="grayscale_lung_images",
                outputs="normalized_lung_images",
                name="normalize_lung_images_node",
                tags=["preprocessing"],
            ),
            node(
                func=group_to_numpy,
                inputs="normalized_lung_images",
                outputs="lung_dataset",
                name="group_lung_images_to_numpy_node",
                tags=["preprocessing"],
            ),
            node(
                func=resize_images,
                inputs="lung_masks",
                outputs="resized_lung_masks",
                name="resize_lung_masks_node",
                tags=["preprocessing"],
            ),
            node(
                func=convert_to_grayscale,
                inputs="resized_lung_masks",
                outputs="grayscale_lung_masks",
                name="grayscale_lung_masks_node",
                tags=["preprocessing"],
            ),
            node(
                func=normalize_images,
                inputs="grayscale_lung_masks",
                outputs="normalized_lung_masks",
                name="normalize_lung_masks_node",
                tags=["preprocessing"],
            ),
            node(
                func=group_to_numpy,
                inputs="normalized_lung_masks",
                outputs="lung_mask_dataset",
                name="group_lung_masks_to_numpy_node",
                tags=["preprocessing"],
            ),
            node(
                func=feature_target_split,
                inputs=["lung_dataset", "lung_mask_dataset", "params:split_ratio"],
                outputs="train_test_dataset",
                name="train_test_split_node",
                tags=["preprocessing"],
            ),
            node(
                func=train_model,
                inputs=["train_test_dataset", "params:model_params"],
                outputs="model",
                name="train_model_node",
                tags=["training"],
            ),
        ]
    )
