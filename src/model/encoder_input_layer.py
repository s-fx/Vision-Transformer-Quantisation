import torch
import torch.nn as nn
from src.model.patch_embedding import PatchEmbedding


class EncoderInputLayer(nn.Module):
    """ Adds positional encoding and a CLS Token which serves as observer neuron
    """
    def __init__(self,
                 img_channels: int,
                 patch_size: int,
                 image_size: int,
                 embed_dim: int,
                 input_dropout_rate: float):
        super().__init__()
        # Create patch embeddings
        self.patch_embeddings = PatchEmbedding(image_size, img_channels, patch_size, embed_dim)

        # Learnabel CLS token
        self.cls_token = nn.Parameter(
            torch.zeros(1,1, embed_dim),
            requires_grad=True
        )

        # Number of patches
        num_patches = self.patch_embeddings.num_patches
        num_positions = num_patches + 1 # + CLS

        # Learnabel positional embedding
        self.positional_embeddings = nn.Parameter(
            torch.zeros(1, num_positions, embed_dim),
            requires_grad=True
        )

        # Initialisation
        nn.init.trunc_normal_(self.positional_embeddings, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.dropout = nn.Dropout(p=input_dropout_rate)


    def forward(self, x):
        batch_size = x.shape[0]

        # Patch embedding
        patch_embeddings = self.patch_embeddings(x)

        # Repeat CLS token for each image in batch
        cls_token = self.cls_token.expand(batch_size, -1, -1) # (B, 1, embed_dim)

        # Concatenate CLS token + patch embeddings
        tokens = torch.concat((cls_token, patch_embeddings), dim=1) # (B, num_patches + 1, embed_dim)

        # Add positional encoding and apply dropout
        return self.dropout(tokens + self.positional_embeddings)



if __name__ == '__main__':
    # Testing Patch Embedding
    dummy = torch.randn(2,3,224,224) # 2 images RGB 224x224
    encoder_input_layer = EncoderInputLayer(3,16,224,512,0.0)
    output = encoder_input_layer(dummy)
    print(f'Batch Image Shape: {dummy.shape}')
    print(f'Output of Encoder Input Layer Shape: {output.shape}')

