import torch.nn as nn
from torchvision import models

class FineTunedVisionTransformer(nn.Module):
    """
    A fine-tuned Vision Transformer (ViT-B/16) model for custom classification tasks.
    
    This class adapts a pre-trained ViT model by freezing its feature extraction
    layers and replacing the final classification head with a new one tailored
    to the specified number of output classes.
    """
    def __init__(self, num_classes: int = 2):
        """
        Initializes the fine-tuned ViT model.
        
        Args:
            num_classes (int): The number of classes for the new classification head.
        """
        super().__init__()

        # Load the pre-trained ViT-B/16 model with ImageNet-1K weights.
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        # Freeze all parameters in the base model to prevent them from being updated
        # during training.
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the classifier head.
        #    - First, get the number of input features from the original head.
        #    - Then, create a new linear layer for our specific number of classes.
        original_head = self.model.heads.head
        num_in_features = original_head.in_features
        
        self.model.heads.head = nn.Linear(in_features=num_in_features, out_features=num_classes)
        
        print(f"Model ready. Base layers frozen. Classifier head adapted for {num_classes} classes.")

    def forward(self, x):
        """
        Performs the forward pass of the model.
        
        Args:
            x (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The output logits from the classifier head.
        """
        return self.model(x)