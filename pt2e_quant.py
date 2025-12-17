import torch
from torch.utils.data import DataLoader, TensorDataset
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

from src.quant.utils import get_model_size, calc_model_size_mb, get_model_size_bytes
from src.model.vit import VisionTransformer, load_model



def pt2e_quantize_deprecate(model, calibration_loader, example_input, device="cuda"):

    model = model.to(device)
    model.eval()

    # 1. Export the model (replaces FX tracing)
    ep = torch.export.export(model, (example_input.to(device),))

    # 2. Choose qconfig for GPU
    qconfig = get_default_pt2e_qconfig("cuda")

    # 3. Prepare model for calibration
    prepared = prepare_pt2e(ep, qconfig)

    # 4. Calibration pass
    with torch.no_grad():
        for batch in calibration_loader:
            images = batch[0].to(device)
            prepared(images)

    # 5. Convert to quantized graph
    quantized = convert_pt2e(prepared)

    return quantized

def pt2e_quantize(model, example_input, calibration_loader, device="cuda"):
    import torch
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torchao.quantization.pt2e.quantizer.composable_quantizer import ComposableQuantizer
    from torchao.quantization.pt2e.quantizer.embedding_quantizer import EmbeddingQuantizer
    from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import X86InductorQuantizer

    model = model.to(device)
    example_input = example_input.to(device)

    # --------------------------------------------
    # 1) Export model to FX Graph (REQUIRED!)
    # --------------------------------------------
    #exported = torch.export.export(
    #    model,
    #    (example_input,),   # must be a tuple
    #)
    exported = torch.export.export(
    model,
    (example_input,),
    )

    # --------------------------------------------
    # 2) Build composed quantizer
    # --------------------------------------------
    quantizer = ComposableQuantizer([
        X86InductorQuantizer(),
        EmbeddingQuantizer(),
    ])

    # --------------------------------------------
    # 3) Prepare
    # --------------------------------------------
    graph_module = exported.module()
    #ep = exported.graph_module
    prepared = prepare_pt2e(graph_module, quantizer)

    # --------------------------------------------
    # 4) Calibration
    # --------------------------------------------
    from torchao.quantization.pt2e import move_exported_model_to_eval
    move_exported_model_to_eval(prepared)
    with torch.no_grad():
        for batch in calibration_loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device)
            prepared(images)

    # --------------------------------------------
    # 5) Convert
    # --------------------------------------------
    quantized_model = convert_pt2e(prepared)

    return quantized_model


if __name__ == "__main__":

    params_path = "./runs/run7_retino_224/params.json"   # if you use load_model
    ckpt_path = "./runs/run7_retino_224/epoch_100_model.pth"  # path to checkpoint used by load_model
    device = "cuda"

    model, _ = load_model(params_path, ckpt_path, device)

    get_model_size_bytes(model)

    # Example input
    example_input = torch.randn(1, 3, 224, 224)

    # Dummy calibration data
    calib_data = torch.randn(64, 3, 224, 224)
    calib_loader = DataLoader(TensorDataset(calib_data), batch_size=1)

    quantized_model = pt2e_quantize(
        model,
        calibration_loader=calib_loader,
        example_input=example_input,
    )

    # Test quantized inference
    out = quantized_model(example_input.cuda())
    print("Quantized output:", out)
    get_model_size_bytes(quantized_model)
    #quantized_model.save("vit_pt2e_quantized_ep.pt2")

