"""
Create roofline model with ncu
run with:
sudo /usr/local/cuda-12.6/bin/ncu \
        --set full --target-processes all \
        --export vit_forward /home/s-fx/venv/vit/bin/python roofline.py
"""

import torch
from src.model.vit import VisionTransformer, load_model

from torch.profiler import profile, ProfilerActivity, record_function
print(torch.__version__)

if __name__ == '__main__':
    # Load Model
    params_path = "./runs/run7_retino_224/params.json"   # if you use load_model
    ckpt_path = "./runs/run7_retino_224/epoch_100_model.pth"  # path to checkpoint used by load_model
    device = "cuda"
    image = torch.randn(1, 3, 224, 224, device=device)

    model, _ = load_model(params_path, ckpt_path, device)
    model = model.to(torch.bfloat16)
    image = image.to(torch.bfloat16)

    model_c = torch.compile(model, mode='max-autotune')

    # Create roofline model of base model, measuring one forward pass
    # Warm-Up
    for _ in range(5):
        _ = model_c(image)



    with profile(
        activities=[ProfilerActivity.CUDA],
        #schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        record_shapes=True,
        profile_memory=True,
        with_flops=True
    ) as prof:
        #with record_function['model_inference']:
        model_c(image)
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))


        #torch.cuda.synchronize()
        #torch.cuda.profiler.start()
        #with torch.no_grad():
        #    _ = model_c(image)
        #torch.cuda.synchronize()
        #torch.cuda.profiler.stop()

    #print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
