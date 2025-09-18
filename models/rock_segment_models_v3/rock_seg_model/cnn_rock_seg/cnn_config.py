# rock_seg_model/cnn_rock_seg/cnn_config.py
CNN_CONFIG = {
    "img_channels": 1,
    "num_classes": 3,
    "channels": [16, 32, 64],   # filtros por camada
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "output_shape": (128, 128)  # saída final da segmentação
}
