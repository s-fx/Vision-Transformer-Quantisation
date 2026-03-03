import torch

print('Per Channel asymmetric quantisation')

torch.manual_seed(42)

weights = 2 * torch.rand((3,3)) -1
print(weights)

quantized_dtype = torch.int8

W_min = weights.min(dim=-1).values
W_max = weights.max(dim=-1).values
print(f'W_min {W_min}')
print(f'W_max {W_max}')

Q_min = torch.iinfo(quantized_dtype).min
Q_max = torch.iinfo(quantized_dtype).max
print(f'Q_min {Q_min}')
print(f'Q_max {Q_max}')

# Scale factor
S = (W_max - W_min) / (Q_max - Q_min)
S = torch.clamp(S, min=1e-8) # Scale can be 0
print(f'Scale {S}')

# Zero Point
Z = torch.round(Q_min - (W_min/S)).to(quantized_dtype)
print(f'Zero Point {Z}')

# Quantized Weights
Q = (weights / S.unsqueeze(1)) + Z.unsqueeze(1)
Q = torch.clamp(torch.round(Q), Q_min, Q_max).to(quantized_dtype)
print(f'Quantized Weights: {Q}')

# Dequantize weights
dequantized_weight = S.unsqueeze(1) * (Q.to(torch.float32) - Z.unsqueeze(1))
print(f'Dequantized Weigths: {dequantized_weight}')

