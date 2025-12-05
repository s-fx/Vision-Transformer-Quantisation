import torch
import torch.nn as nn
import json

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




def load_model(params_path, ckpt_path, device):
    with open(params_path, 'r') as f:
        p = json.load(f)

    # Load Model
    model = VisionTransformer(in_channels=p['in_channels'],
                              image_size=p['image_size'],
                              patch_size=p['patch_size'],
                              number_of_encoder=p['number_of_encoder'],
                              embeddings=p['embeddings'],
                              d_ff_scale=p['d_ff_scale'],
                              num_heads=p['num_heads'],
                              input_dropout_rate=p['input_dropout_rate'],
                              attention_dropout_rate=p['attention_dropout_rate'],
                              feed_forward_dropout_rate=p['feed_forward_dropout_rate'],
                              number_of_classes=p['number_of_classes']
                             )

    # Load weights and loss dict
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss_dict = checkpoint.get('loss_dict', {})

    model.to(device)
    model.eval()
    return model, loss_dict



if __name__ == '__main__':
    # Testing Patch Embedding
    dummy = torch.randn(2,3,224,224) # 2 images RGB 224x224

    vit = VisionTransformer(3,224,16,3,512,4,4,0.0,0.0,0.0,2)
    out = vit(dummy)
    print(out.shape)

