import torch
import torch.nn as nn

from src.model.encoder import Encoder
from src.model.encoder_input_layer import EncoderInputLayer


class VisionTransformer(nn.Module):
    def __init__(self, in_channels, image_size, patch_size, number_of_encoder,
                 embeddings, d_ff_scale, num_heads, input_dropout_rate,
                 attention_dropout_rate, feed_forward_dropout_rate, number_of_classes):
        super().__init__()
        self.input_layer = EncoderInputLayer(in_channels, patch_size, image_size, embeddings, input_dropout_rate)
        self.encoder_stack = Encoder(number_of_encoder, embeddings, d_ff_scale, num_heads,
                                     attention_dropout_rate, feed_forward_dropout_rate)
        self.classification_head = nn.Sequential(
            nn.LayerNorm([embeddings]),
            nn.Linear(embeddings, number_of_classes)
        )

    def forward(self, x):
        return self.classification_head(self.encoder_stack(self.input_layer(x))[:, 0, :])



if __name__ == '__main__':
    # Testing Patch Embedding
    dummy = torch.randn(2,3,224,224) # 2 images RGB 224x224

    vit = VisionTransformer(3,224,16,3,512,4,4,0.0,0.0,0.0,2)
    out = vit(dummy)
    print(out.shape)

