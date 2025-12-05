import os
import copy
import torch
import numpy as np

from src.dataset.dataset import load_dataset
from src.model.vit import load_model

from torchao.quantization import Int4WeightOnlyConfig, quantize_
from torchao.utils import (
    benchmark_model,
    unwrap_tensor_subclass,
)



# Actual RAM footprint of live model
def get_model_size(model):
    dtypes = []
    # Check weights dtype
    for name, param in model.named_parameters():
        dtypes.append(param.dtype)
    print(f'Layer weights dtype: {set(dtypes)}')

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 / 1024
    print('model size: {:.3f}MB'.format(size_all_mb))


# Size on disk of serialized state_dict
def calc_model_size_mb(ckpt_path):
    float32_model_size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
    print('float32 model size: %.2f MB' % float32_model_size_mb)



def main():
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    root = './runs/run7_retino_224/'
    data_root = '/home/s-fx/fun/datasets/retinopathy-full-ds'
    params_path = os.path.join(root, 'params.json')
    ckpt_path = os.path.join(root, 'epoch_100_model.pth')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'[INFO] Device in use: {device}')
    model, loss_dict = load_model(params_path, ckpt_path, device)
    #model = model.eval().to(torch.bfloat16).to(device)
    model_float32 = copy.deepcopy(model)

    # Calculate model size
    get_model_size(model_float32)
    calc_model_size_mb(ckpt_path)

    # Quantize Model
    #quantize_(model, Int4WeightOnlyConfig(group_size=32))
    #torch.save(model, "/tmp/int4_model.pt")
    #int4_model_size_mb = os.path.getsize("/tmp/int4_model.pt") / 1024 / 1024
    #print("int4 model size: %.2f MB" % int4_model_size_mb)

    # Calc speedup
    num_runs = 100
    torch._dynamo.reset()
    example_inputs = (torch.randn((1,3,224,224), dtype=torch.float32, device="cuda"),)
    f32_time = benchmark_model(model_float32, num_runs, example_inputs)
    #int4_time = benchmark_model(model, num_runs, example_inputs)

    print("f32 mean time: %0.3f ms" % f32_time)
    #print("int4 mean time: %0.3f ms" % int4_time)
    #print("speedup: %0.1fx" % (f32_time / int4_time))








    #test_dataloader = load_dataset(data_root)


if __name__ == '__main__':
    main()
