"""
Using torchao quantization without backend specific optimization
"""

import torch
from torchao.quantization.quant_api import quantize_, Int8DynamicActivationInt8WeightConfig
from torchao.utils import unwrap_tensor_subclass
from torch.utils.benchmark import Timer

from src.model.vit import VisionTransformer, load_model
from src.dataset.dataset import load_dataset


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

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        model_c(image)

    print(prof.key_averages().table(sort_by="cuda_time_total"))


    ###################################################
    # Quantization
    del model_c, model, image
    model, _ = load_model(params_path, ckpt_path, device)
    model = model.to(torch.bfloat16)
    image = torch.randn(1, 3, 224, 224, device=device)
    image = image.to(torch.bfloat16)
    quantize_(model, Int8DynamicActivationInt8WeightConfig())
    model_c = torch.compile(model, mode='max-autotune')
    quant_res = benchmark(model_c, image)
    print(f"bf16 compiled runtime of the quantized model is {quant_res['time']:0.2f}ms : peak memory {quant_res['memory']:0.2f}GB \
: reserved memory {quant_res['reserved']:0.2f}GB")
    # Slower because every forward pass:
    #   scan activation tensor
    #   compute scale / zero point
    #   quantize activation into int8
    #   run matmul
    #   dequantize output back to bf16
    # Memory usage goes down:
    #   weights in int8 -> 4x smaller

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        model_c(image)

    print(prof.key_averages().table(sort_by="cuda_time_total"))



    # Quantization improved
    # Even though we are doing a quantized matmul, such as ``int8 x int8``,
    #    the result of the multiplication gets stored in an int32 tensor
    #    which is twice the size of the result from the non-quantized model.
    #    If we can avoid creating this int32 tensor, our memory usage will improve a lot.
    # We can fix it by fusing the integer matmul with the subsequent rescale
    # operation since the final output will be bf16, if we immediately convert
    # the int32 tensor to bf16 and instead store that we’ll get better
    # performance in terms of both runtime and memory.
    #
    # The way to do this, is to enable the option
    # ``force_fuse_int_mm_with_mul`` in the inductor config.

    del model_c, model, image
    model, _ = load_model(params_path, ckpt_path, device)
    model = model.to(torch.bfloat16)
    image = torch.randn(1, 3, 224, 224, device=device)
    image = image.to(torch.bfloat16)
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    quantize_(model, Int8DynamicActivationInt8WeightConfig())
    model_c = torch.compile(model, mode='max-autotune')
    quant_res = benchmark(model_c, image)
    print(f"bf16 compiled runtime of the quantized model is {quant_res['time']:0.2f}ms : peak memory {quant_res['memory']:0.2f}GB \
: reserved memory {quant_res['reserved']:0.2f}GB")











##########################################################################################################################################
"""
    # Quantization final
    del model_c, model, image
    model, _ = load_model(params_path, ckpt_path, device)
    model = model.to(torch.bfloat16)
    image = torch.randn(1, 3, 224, 224, device=device)
    image = image.to(torch.bfloat16)
    torch._inductor.config.epilogue_fusion = False
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.coordinate_descent_check_all_directions = True
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    quantize_(model, Int8DynamicActivationInt8WeightConfig())
    model_c = torch.compile(model, mode='max-autotune')
    quant_res = benchmark(model_c, image)
    print(f"bf16 compiled runtime of the quantized model is {quant_res['time']:0.2f}ms : peak memory {quant_res['memory']:0.2f}GB \
: reserved memory {quant_res['reserved']:0.2f}GB")

    # Create roofline model of base model, measuring one forward pass
    # Warm-Up
    for _ in range(5):
        _ = model_c(image)

    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    with torch.no_grad():
        _ = model_c(image)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
"""


