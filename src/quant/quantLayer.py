"""
Custom 8-bit quantization Layer
Static asymetric quantization
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.vit import load_model


# Per channel fp32 -> int8 asymmetric quantization
class QuantizedLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()
        self.register_buffer('weight', torch.randint(-128, 127, (out_features, in_features)).to(torch.int8))
        self.register_buffer('scale', torch.randn((out_features), dtype=dtype))
        self.register_buffer("zero_point", torch.randn(out_features, dtype=dtype))
        if bias:
            self.register_buffer("bias", torch.randn(out_features, dtype=dtype))
        else:
            self.bias = None


    def quantize(self, weight):
        weight_fp32 = weight.clone().to(torch.float32)

        Q_min = torch.iinfo(torch.int8).min
        Q_max = torch.iinfo(torch.int8).max

        # Calculate per channel scale (scale per row)
        W_max = weight_fp32.max(dim=-1).values
        W_min = weight_fp32.min(dim=-1).values
        scale = (W_max - W_min) / (Q_max - Q_min)
        # scale can be 0
        scale = torch.clamp(scale, min=1e-8)

        # Calculate zero point per channel
        zero_point = torch.round(Q_min - (W_min / scale)).to(torch.int8)

        # Calculate quantized tensor
        quantized_weight = (weight_fp32 / scale.unsqueeze(1)) + zero_point.unsqueeze(1)
        quantized_weight = torch.clamp(torch.round(quantized_weight), Q_min, Q_max).to(torch.int8)

        self.weight.copy_(quantized_weight)
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)


    def forward(self, input):
        weights_fp32 = (self.weight.to(input.dtype) - self.zero_point.unsqueeze(1)) * self.scale.unsqueeze(1)
        output = F.linear(input, weights_fp32)
        if self.bias is not None:
            output = output + self.bias
        return output


def replace_linear_layers(model, quantizeLayer_class, exclude_list, quantized=True):
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and not any([x == name for x in exclude_list]):
            old_bias = child.bias
            old_weight = child.weight
            in_features = child.in_features
            out_features = child.out_features

            # This is the stage where we'll initialize the quantizer class with the in_features, out_features, bias and dtype.
            # The base_model parameters values are given to the quantizer class parameters.
            quantizer_layer = quantizeLayer_class(in_features, out_features, old_bias is not None, old_weight.dtype)

            # Quantize weights
            quantizer_layer.quantize(old_weight.data)

            # Copy bias
            if child.bias is not None:
                quantizer_layer.bias.copy_(old_bias.data)
            setattr(model, name, quantizer_layer)

            # After the quantizer class is initialized, The replacement takes place as below.
            #setattr(model, name, quantizer_layer)

            # Now that after replacement, base_model linear layer is now a quantizer layer.
            # We can now call the quantize_layers quantize function to quantize the old_weights 
            # of FP16 new quantized weights of int8 type.
            #if quantized:
            #    getattr(model, name).quantize(old_weight)

            # If bias is not none, we'll also update bias with the base model bias value
            #if old_bias is not None:
            #    getattr(model, name).bias = old_bias

        # If the base model child instance has further sub-components with linear layers, 
        # we'll have to quantize them by calling the replace_linear_layer function with the child as base_model now.
        # This will replace all the linear layers with quantized layers that are under the child sub-section.
        else:
            replace_linear_layers(child, quantizeLayer_class, exclude_list, quantized=quantized)



def main():
    root = './runs/run8_cifar_224'
    params_path = os.path.join(root, 'params.json')
    ckpt_path = os.path.join(root, 'epoch_100_model.pth')
    device = 'cuda'
    model, loss_dict = load_model(params_path, ckpt_path, device)
    print(model)

    print(f'\nQuantization')
    replace_linear_layers(model, QuantizedLinearLayer, [''], quantized=True)
    print(model)
    for m in model.modules():
        if isinstance(m, QuantizedLinearLayer):
            print("Found quantized layer:", m)




if __name__ == '__main__':

    torch.manual_seed(0)

    in_features = 16
    out_features = 8
    batch_size = 4

    # Reference FP32 layer
    fp32_layer = nn.Linear(in_features, out_features, bias=True)

    # Quantized layer
    q_layer = QuantizedLinearLayer(in_features, out_features, bias=True)

    # Quantize FP32 weights
    q_layer.quantize(fp32_layer.weight.data)
    q_layer.bias.copy_(fp32_layer.bias.data)

    # Input
    x = torch.randn(batch_size, in_features)

    # Outputs
    y_fp32 = fp32_layer(x)
    y_q = q_layer(x)

    # Error
    print("Max abs error:", (y_fp32 - y_q).abs().max().item())
    print("Mean abs error:", (y_fp32 - y_q).abs().mean().item())

    # eine Zeile = patch
    # channel quantization would quantize each row of the weights...
    # asymetric quantization
    weights = torch.rand((4,4))
    print(weights)
    print(weights.abs().max(dim=-1))

    print(q_layer.weight.dtype)              # torch.int8
    print(q_layer.weight.min(), q_layer.weight.max())  # within [-128, 127]
    print(q_layer.scale.shape)               # [out_features]
    print(q_layer.zero_point.shape)          # [out_features]

    # ensure each output channel has its own scale
    print("Scales:", q_layer.scale)
    print("Are scales different?", torch.any(q_layer.scale != q_layer.scale[0]))

    # Signal to quantization noise ratio
    noise = y_fp32 - y_q
    sqnr = 10 * torch.log10(y_fp32.pow(2).mean() / noise.pow(2).mean())
    print("SQNR (dB):", sqnr.item())

    """
    quantized_dtype = torch.int8

    W_min = weights.min().item()
    W_max = weights.max().item()
    print(f'W_min {W_min}')
    print(f'W_max {W_max}')

    Q_min = torch.iinfo(quantized_dtype).min
    Q_max = torch.iinfo(quantized_dtype).max
    print(f'Q_min {Q_min}')
    print(f'Q_max {Q_max}')

    # Scale factor
    S = (W_max - W_min) / (Q_max - Q_min)
    print(f'Scale {S}')

    # Zero Point
    Z = int(round(Q_min - (W_min/S)))
    print(f'Zero Point {Z}')

    # Quantized Weights
    Q = (weights / S) + Z
    Q = torch.clamp(torch.round(Q), Q_min, Q_max)
    Q = Q.to(quantized_dtype)
    print(Q)

    # Dequantize weights
    dequantized_weight = S * (Q.to(torch.float32) - Z)
    print(dequantized_weight)
    """
