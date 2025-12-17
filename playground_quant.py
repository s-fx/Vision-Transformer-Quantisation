import torch
from torchao.quantization import get_default_pt2e_quantizer
import sys


quantizer = get_default_pt2e_quantizer("cuda")
print(quantizer)
sys.exit()





class M(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.linear = torch.nn.Linear(5, 10)

   def forward(self, x):
      return self.linear(x)


example_inputs = (torch.randn(1, 5),)
m = M().eval()
print(f'Float Model: {m}')
print(30*'-')

# Step 1. program capture
# This is available for pytorch 2.6+, for more details on lower pytorch versions
# please check `Export the model with torch.export` section
m = torch.export.export(m, example_inputs).module()

# we get a model with aten ops
print(type(m))
print(30*'-')
print(f'Model with aten ops: {m}')
print(30*'-')
m.print_readable()
print(30*'-')

# Step 2. quantization
from torchao.quantization.pt2e.quantize_pt2e import (
  prepare_pt2e,
  convert_pt2e,
)

# install executorch: `pip install executorch`
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
  get_symmetric_quantization_config,
  XNNPACKQuantizer,
)
# backend developer will write their own Quantizer and expose methods to allow
# users to express how they
# want the model to be quantized
quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
m = prepare_pt2e(m, quantizer)

# calibration omitted

m = convert_pt2e(m)
# we have a model with aten ops doing integer computations when possible
