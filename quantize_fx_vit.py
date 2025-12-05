# quantize_fx_vit.py
import os
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

# import your model loader (adjust import path if needed)
from src.model.vit import VisionTransformer, load_model
from src.model.encoder import LayerNormalization  # your custom LayerNormalization


from torchao.quantization.pt2e import (
    prepare_pt2e,
    convert_pt2e,
    get_default_pt2e_qconfig
)


# ---------------------------
# Utility functions
# ---------------------------
def get_model_size_bytes(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return param_size + buffer_size

def print_model_stats(tag, model):
    bytes_size = get_model_size_bytes(model)
    print(f"{tag} - total parameters+buffers: {bytes_size / 1024 / 1024:.3f} MB")

def benchmark_inference(model, example_input, runs=50, warmup=10, device="cpu"):
    model.to(device)
    model.eval()
    times = []
    # warmup
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(example_input.to(device))
        # timed runs
        start = time.time()
        for _ in range(runs):
            _ = model(example_input.to(device))
        duration = time.time() - start
    print(f"Avg inference time over {runs} runs: {duration / runs:.6f} s")
    return duration / runs

# NEW WAY
def pt2e_quantize(model, calibration_loader, example_input):

    # 1. Export the model
    ep = torch.export.export(model, (example_input,))

    # 2. QConfig
    qconfig = get_default_pt2e_qconfig("x86")

    # 3. Prepare
    prepared = prepare_pt2e(ep, qconfig)

    # 4. Calibration
    with torch.no_grad():
        for images, _ in calibration_loader:
            prepared(images)

    # 5. Convert
    quantized = convert_pt2e(prepared)

    return quantized


# ---------------------------
# Main FX quantization flow
# ---------------------------
def fx_post_training_quantize(model,
                              calibration_loader=None,
                              example_inputs=None,
                              qconfig_backend="fbgemm",
                              device="cpu"):
    # 1) Move model to cpu and eval (FX currently prefers CPU flow)
    model.to("cpu")
    model.eval()

    # 2) Choose qconfig
    default_qconfig = get_default_qconfig(qconfig_backend)
    qconfig_dict = {"": default_qconfig}  # default for everything

    # 3) Prevent quantization of custom LayerNormalization (keep in FP32)
    #    You can also set specific modules to qconfig = None
    for name, mod in model.named_modules():
        if isinstance(mod, LayerNormalization):
            mod.qconfig = None

    # 4) Prepare FX. You MUST provide example_inputs for correct tracing.
    if example_inputs is None:
        # fallback to a single random example (B=1,3,224,224)
        example_inputs = (torch.randn(1, 3, 224, 224),)
    prepared = prepare_fx(model, qconfig_dict, example_inputs)

    # 5) Calibration: run representative data through 'prepared' to collect stats
    if calibration_loader is None:
        print("No calibration_loader provided, running 50 random samples for calibration.")
        with torch.inference_mode():
            for _ in range(50):
                inp = torch.randn(*example_inputs[0].shape)
                prepared(inp)
    else:
        print("Calibrating with provided dataset...")
        with torch.inference_mode():
            for batch in calibration_loader:
                # support DataLoader that yields (x, y) or x
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                prepared(x)

    # 6) Convert to quantized model
    quantized = convert_fx(prepared)

    # 7) Move to desired device (CPU recommended for int8 with fbgemm)
    quantized.to(device)
    quantized.eval()
    return quantized

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # ---- configure these ----
    params_path = "./runs/run7_retino_224/params.json"   # if you use load_model
    ckpt_path = "./runs/run7_retino_224/epoch_100_model.pth"  # path to checkpoint used by load_model
    device = "cpu"
    # -------------------------

    # 1) Load your trained model (option A: via your load_model helper)
    # If you use load_model, use it. Otherwise instantiate VisionTransformer directly.
    model, _ = load_model(params_path, ckpt_path, device)  # uncomment if applicable

    # OR: instantiate and load weights manually:
    # Adjust these hyperparams to match the one you trained
    #model = VisionTransformer(
    #    in_channels=3, image_size=224, patch_size=16,
    #    number_of_encoder=3, embeddings=512, d_ff_scale=4,
    #    num_heads=4, input_dropout_rate=0.0,
    #    attention_dropout_rate=0.0, feed_forward_dropout_rate=0.0,
    #    number_of_classes=10
    #)

    # Optionally load a checkpoint (example):
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # If your checkpoint saved state_dict under 'model_state_dict':
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print("Loaded checkpoint:", ckpt_path)

    # 2) Optional: create a calibration DataLoader (recommended to use real images)
    # Example: use 256 random images as placeholder
    calib_inputs = torch.randn(256, 3, 224, 224)
    calib_loader = DataLoader(TensorDataset(calib_inputs), batch_size=8)

    # 3) Example input for tracing & benchmarking
    example_input = (torch.randn(1, 3, 224, 224),)

    # Print FP32 stats & benchmark
    print_model_stats("FP32 (raw)", model)
    print("FP32 benchmark:")
    benchmark_inference(model, example_input[0], runs=20, warmup=5, device="cpu")

    # 4) Run FX PTQ
    #quantized_model = fx_post_training_quantize(
    #    model,
    #    calibration_loader=calib_loader,
    #    example_inputs=example_input,
    #    qconfig_backend="fbgemm",
    #    device="cpu"
    #)
    quantized = pt2e_quantize(model, calib_loader, example_input)

    print("Quantized output:", quantized(example_input))
    torch.save(quantized, "vit_pt2e_quantized.pt")

    # 5) Print quantized stats & benchmark
    print_model_stats("INT8 (raw)", quantized)
    print("INT8 benchmark:")
    benchmark_inference(quantized, example_input[0], runs=20, warmup=5, device="cpu")

    # 6) Save quantized model (use torch.save)
    #torch.save({"model_state_dict": quantized_model.state_dict()}, "vit_quantized_fx.pth")
    #print("Saved quantized checkpoint to vit_quantized_fx.pth")

