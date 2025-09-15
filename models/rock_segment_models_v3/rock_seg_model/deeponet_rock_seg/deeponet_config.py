# deeponet_config.py
DEEPONET_CONFIG = {
    "img_channels": 1,
    "num_classes": 3,
    "branch_conv_channels": [16, 32],
    "branch_kernel_sizes": [3, 3],
    "branch_strides": [1, 1],
    "trunk_conv_channels": [16, 32],
    "trunk_kernel_sizes": [3, 3],
    "trunk_strides": [1, 1],
    "hidden_dim": 64,
    "output_shape": (128, 128)
}
