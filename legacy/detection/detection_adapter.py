import torch
import torch.nn as nn


class DetectionAdapter(nn.Module):
    """
    Adapter module to reshape LLM hidden states for detection tasks.

    Consists of one or more residual MLP bottleneck blocks.

    Args:
        hidden_size: Dimension of the input and output features.
        bottleneck: Dimension of the adapter's hidden bottleneck.
        num_layers: Number of sequential adapter blocks to stack.
    """

    def __init__(self, hidden_size: int, bottleneck: int, num_layers: int = 1) -> None:
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, bottleneck))
            layers.append(nn.SiLU())
            layers.append(nn.Linear(bottleneck, hidden_size))
        self.adapter = nn.Sequential(*layers)

        # Initialize adapter weights
        for m in self.adapter:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection: add the adapter output to the original features
        return x + self.adapter(x)
