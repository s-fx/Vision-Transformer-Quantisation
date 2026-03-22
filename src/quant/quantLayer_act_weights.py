"""
Custom 8-bit quantization Layer
Weights : static  symmetric  per-channel quantization with min-max clipping
Activations: dynamic asymmetric per-channel quantization with min-max clipping
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# Ensure the project root (ViT-Classificator/) is on the path regardless of
# where the script is launched from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.model.vit import load_model


# Weights : static  symmetric  per-channel (per output row) quantization
# Activations: dynamic asymmetric per-channel (per input feature) quantization
class QuantizedLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()
        self.register_buffer('weight', torch.randint(-128, 127, (out_features, in_features)).to(torch.int8))
        self.register_buffer('scale', torch.randn((out_features), dtype=dtype))
        # No zero_point needed for symmetric weight quantization (always 0)
        if bias:
            self.register_buffer("bias", torch.randn(out_features, dtype=dtype))
        else:
            self.bias = None


    def quantize(self, weight):
        weight_fp32 = weight.clone().to(torch.float32)

        # Symmetric quantization uses [-127, 127] (not -128) to keep range symmetric
        Q_min = -127
        Q_max = 127

        # Min-max clipping: find the absolute max per channel (row)
        # This maps the largest magnitude value to ±127
        W_abs_max = torch.max(weight_fp32.abs().max(dim=-1).values, torch.tensor(1e-8))

        # Symmetric scale: no zero_point needed, range is centred at 0
        scale = W_abs_max / Q_max

        # Quantize: divide by scale, round, then clip to [-127, 127]
        quantized_weight = torch.clamp(torch.round(weight_fp32 / scale.unsqueeze(1)), Q_min, Q_max).to(torch.int8)

        self.weight.copy_(quantized_weight)
        self.scale.copy_(scale)


    def quantize_activations(self, input):
        """
        Dynamic asymmetric per-channel (per input feature) quantization.
        'Per channel' for activations = one scale/zero_point per feature dimension.
        Input shape: (..., in_features)  →  reduce over all dims except the last.
        Returns: (quantized_int8, scale, zero_point)
          scale      : (in_features,)
          zero_point : (in_features,)  dtype int8
        """
        input_fp32 = input.to(torch.float32)

        Q_min = torch.iinfo(torch.int8).min   # -128
        Q_max = torch.iinfo(torch.int8).max   #  127

        # Reduce over every dimension except the feature dim (last dim)
        reduce_dims = list(range(input_fp32.dim() - 1))   # e.g. [0] for (B, C)
        A_min = input_fp32.amin(dim=reduce_dims)           # (in_features,)
        A_max = input_fp32.amax(dim=reduce_dims)           # (in_features,)

        # Asymmetric min-max scale per feature
        scale = (A_max - A_min) / (Q_max - Q_min)
        scale = torch.clamp(scale, min=1e-8)               # avoid division by zero

        # Zero point maps Q_min to A_min
        zero_point = torch.clamp(
            torch.round(Q_min - A_min / scale), Q_min, Q_max
        ).to(torch.int8)

        # Quantize and clamp to int8 range
        quantized = torch.clamp(
            torch.round(input_fp32 / scale + zero_point.to(torch.float32)),
            Q_min, Q_max
        ).to(torch.int8)

        return quantized, scale, zero_point


    def forward(self, input):
        # --- Quantize activations dynamically (asymmetric, per-channel) ---
        act_q, act_scale, act_zero_point = self.quantize_activations(input)

        # Dequantize activations back to fp32 for the linear op
        act_deq = (act_q.to(input.dtype) - act_zero_point.to(input.dtype)) * act_scale

        # --- Dequantize weights (symmetric, per-channel) ---
        weights_fp32 = self.weight.to(input.dtype) * self.scale.unsqueeze(1)

        output = F.linear(act_deq, weights_fp32)
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

    # Error (weights + activations both quantized)
    print("Max abs error:", (y_fp32 - y_q).abs().max().item())
    print("Mean abs error:", (y_fp32 - y_q).abs().mean().item())

    # --- Inspect activation quantization independently ---
    act_q, act_scale, act_zp = q_layer.quantize_activations(x)
    act_deq = (act_q.to(torch.float32) - act_zp.to(torch.float32)) * act_scale
    act_err = (x - act_deq).abs()
    print("\nActivation quantization (dynamic asymmetric per-channel):")
    print("  act_scale shape   :", act_scale.shape)    # (in_features,)
    print("  act_zp shape      :", act_zp.shape)       # (in_features,)
    print("  Max act error     :", act_err.max().item())
    print("  Mean act error    :", act_err.mean().item())

    # eine Zeile = patch
    # Symmetric per-channel quantization quantizes each row of the weights
    # using abs-max as the clipping range
    weights = torch.rand((4,4))
    print(weights)
    print(weights.abs().max(dim=-1))

    print(q_layer.weight.dtype)              # torch.int8
    print(q_layer.weight.min(), q_layer.weight.max())  # within [-127, 127]
    print(q_layer.scale.shape)               # [out_features]
    # No zero_point stored for weights (symmetric)

    # ensure each output channel has its own scale
    print("Weight scales:", q_layer.scale)
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
