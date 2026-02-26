import os
import torch
from torch.utils.benchmark import Timer

def get_model_size(model):
    """
        Get the actual RAM footprint of live model
    """
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


def calc_model_size_mb(ckpt_path):
    """
        Size on disk of serialized state_dict
    """
    float32_model_size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
    print('float32 model size: %.2f MB' % float32_model_size_mb)


def get_model_size_bytes(model):
    """
        Get the model size in bytes
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    #print(f"Total parameters+buffers: {bytes_size / 1024 / 1024:.3f} MB")
    return param_size + buffer_size


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

