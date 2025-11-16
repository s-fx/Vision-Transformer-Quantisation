import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """ Creates patches of the image and makes a linear projection
        of each patch into an embedding vector of dimension D (hidden size of
        the transformer)
        We need images of size nxn
    """
    def __init__(self,
                 img_size: int,
                 img_channels: int,
                 patch_size: int,
                 embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels=img_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, x):
        """
        x: (B,C,H,W)
        returns: (B,num_patches,embed_dim)
        """
        image_dimension = x.shape[-1]
        assert image_dimension % self.patch_size == 0, \
                f'Given image dimension {image_dimension} is not divisible by patch_size {patch_size}'
        if len(x.shape) == 3:
            x = x.unsqueeze(0) # add batch dim if input (C,H,W)

        x = self.proj(x) # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2) # flatten height and width: (B, embed_dim, num_patches)
        x = x.permute(0,2,1) # (B, num_patches, embed_dim)
        return x



if __name__ == '__main__':

    # Testing Patch Embedding
    dummy = torch.randn(2,3,224,224) # 2 images RGB 224x224
    patch_embed = PatchEmbedding(img_size=224, img_channels=3, patch_size=16, embed_dim=512)
    output = patch_embed(dummy)
    print(f'Batch Image Shape: {dummy.shape}')
    print(f'Output of Patch Embedding Shape: {output.shape}')



