import torch
import functools
import typing

from models.base_model import BaseModel

class MLP(BaseModel):
    def __init__(
            self,
            input_shape: typing.Union[int, typing.Iterable[int]],
            hidden_channels: typing.Union[int, typing.Iterable[int]],
            n_targets: int = 1,
            norm_layer: typing.Optional[typing.Callable[..., torch.nn.Module]] = torch.nn.BatchNorm1d,
            activation_layer: typing.Optional[typing.Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            activation_output_layer: typing.Optional[typing.Callable[..., torch.nn.Module]] = None,
            bias: bool = True,
            dropout: float = 0.0,
        ):
        super().__init__()
        if isinstance(input_shape, int):
            input_dim = input_shape
        else:
            input_dim = functools.reduce(lambda x, y: x * y, input_shape)

        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]

        layers = [torch.nn.Flatten(1)]
        in_dim = input_dim
        for hidden_dim in hidden_channels:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer())
            if dropout != 0:
              layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, n_targets, bias=bias))
        if activation_output_layer is not None:
            layers.append(activation_output_layer())
        self.model = torch.nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)
    