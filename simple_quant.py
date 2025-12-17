"""
Using torchao quantization without backend specific optimization
"""

import torch
from torchao.quantization.quant_api import quantize_, Int8DynamicActivationInt8WeightConfig
from torchao.utils import unwrap_tensor_subclass
from torch.utils.benchmark import Timer

from src.model.vit import VisionTransformer, load_model


@torch.no_grad()
def benchmark(f, *args, **kwargs):
    """
    Benchmarks GPU inference latency and peak memory.
    Measures memory actively allocated by PyTorch tensors during the benchmark.
    Not total GPU memory reserved + CUDA context + caching allocator + libraries.
    allocated memory -> what tensors currently need
    reserved memory -> what PyTorch asked CUDA for

    Returns:
        Runtime in ms
        Peak Memory in GB
        CUDA reserved memory in GB
    """
    for _ in range(3):
        f(*args, **kwargs)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    res = t0.adaptive_autorange(.03, min_run_time=.2, max_run_time=20)
    return {'time':res.median * 1e3, 'memory': torch.cuda.max_memory_allocated()/1e9, 'reserved': torch.cuda.memory_reserved()/1e9}



if __name__ == '__main__':
    # Load Model
    params_path = "./runs/run7_retino_224/params.json"   # if you use load_model
    ckpt_path = "./runs/run7_retino_224/epoch_100_model.pth"  # path to checkpoint used by load_model
    device = "cuda"

    ################################################################################################################
    # Load Model in Float32
    model, _ = load_model(params_path, ckpt_path, device)
    model.to(device)

    # Dummy Image Input
    image = torch.randn(1, 3, 224, 224, device=device)

    fp32_res = benchmark(model, image)
    print(f"base fp32 runtime of the model is {fp32_res['time']:0.2f}ms : peak memory {fp32_res['memory']:0.2f}GB \
: reserved memory {fp32_res['reserved']:0.2f}GB")


    ################################################################################################################
    # Convert model to bfloat16
    # We can achieve an instant performance boost by converting the model to bfloat16.
    # The reason we opt for bfloat16 over fp16 is due to its dynamic range, which is comparable to
    # that of fp32. Both bfloat16 and fp32 possess 8 exponential bits, whereas fp16 only has 4. This
    # larger dynamic range helps protect us from overflow errors and other issues that can arise
    # when scaling and rescaling tensors due to quantization
    model, _ = load_model(params_path, ckpt_path, device)
    model = model.to(torch.bfloat16)
    image = image.to(torch.bfloat16)
    bf16_res = benchmark(model, image)
    print(f"bf16 runtime of the model is {bf16_res['time']:0.2f}ms : peak memory {bf16_res['memory']:0.2f}GB \
: reserved memory {bf16_res['reserved']:0.2f}GB")


    ################################################################################################################
    # Use torch.compile to further improve performance
    # TorchDynamo -> captures your Python model
    # AOTAutograd -> traces forward graph
    # Inductor    -> generates optimized CUDA code
    # Triton      -> fuses kernels (especially for attention, layernorm, matmul)
    model_c = torch.compile(model, mode='max-autotune')
    comp_res = benchmark(model_c, image)
    print(f"bf16 compiled runtime of the model is {comp_res['time']:0.2f}ms : peak memory {comp_res['memory']:0.2f}GB \
: reserved memory {comp_res['reserved']:0.2f}GB")


    ################################################################################################################

