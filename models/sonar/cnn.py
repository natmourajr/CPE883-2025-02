import typing
import torch
from models.base_model import BaseModel
from models.mlp import MLP

class CNN(BaseModel):

    def __init__(self,
                 input_shape: typing.Iterable[int],
                 conv_n_neurons: typing.List[int],
                 conv_activation: torch.nn.Module = torch.nn.ReLU,
                 conv_pooling: typing.Optional[torch.nn.Module] = torch.nn.MaxPool2d,
                 conv_pooling_size: typing.List[int] = [2,2],
                 conv_dropout: float = 0.5,
                 batch_norm: typing.Optional[torch.nn.Module] = torch.nn.BatchNorm2d,
                 kernel_size: int = 5,
                 padding: int = None,
                 has_class_head = False,
                 hidden_channels = None,
                 n_targets = None,
                 dropout: float = None
                 ):
        super().__init__()

        padding = padding if padding is not None else int((kernel_size-1)/2)

        if len(input_shape) != 3:
            raise UnboundLocalError(f"CNN expects as input an image in the format: \
                                    channel x width x height (current {input_shape})")
        self.has_class_head = has_class_head

        self.input_shape = input_shape

        conv_layers = []
        conv = [self.input_shape[0]]
        conv.extend(conv_n_neurons)

        for i in range(1, len(conv)):
            conv_layers.append(torch.nn.Conv2d(conv[i - 1], conv[i],
                                               kernel_size=kernel_size, padding=padding))
            if batch_norm is not None:
                conv_layers.append(batch_norm(conv[i]))
            if conv_dropout != 0 and i != 0:
                conv_layers.append(torch.nn.Dropout2d(p=conv_dropout))
            conv_layers.append(conv_activation())
            if conv_pooling is not None:
                conv_layers.append(conv_pooling(*conv_pooling_size))

        self.conv_layers = torch.nn.Sequential(*conv_layers)

        test_shape = [1]
        test_shape.extend(input_shape)
        test_tensor = torch.rand(test_shape, dtype=torch.float32)
        device = next(self.parameters()).device
        test_tensor = test_tensor.to(device)
        self.conv_layers = self.conv_layers.to(device)

        test_tensor = self.to_feature_space(test_tensor)
        self.mlp = MLP(
            input_shape=test_tensor.shape,
            hidden_channels=hidden_channels,
            n_targets=n_targets,
            dropout=dropout
        )


    def to_feature_space(self, data: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(data)


    def forward(self, data: torch.Tensor, embeddings=False) -> torch.Tensor:
        data = self.to_feature_space(data)
        
        if embeddings:
            return data
        
        if self.has_class_head:
            data = self.mlp(data)
        
        return data