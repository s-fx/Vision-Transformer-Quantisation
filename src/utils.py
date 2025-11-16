import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def check_loaded_layers(model, checkpoint):
    """
    Prints a summary of which layers were successfully loaded
    from a pretrained checkpoint and which were not.
    Works well with ViTs or any PyTorch model.

    Args:
        model (torch.nn.Module): your model instance
        checkpoint (dict): pretrained state_dict (from torch.load or hub)
    """

    model_dict = model.state_dict()

    matched_keys = []
    mismatched_keys = []
    missing_keys = []
    unexpected_keys = []

    # Identify missing/unexpected
    for k in checkpoint.keys():
        if k not in model_dict:
            unexpected_keys.append(k)
    for k in model_dict.keys():
        if k not in checkpoint:
            missing_keys.append(k)

    # Identify matched/mismatched shapes
    for k, v in checkpoint.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                matched_keys.append(k)
            else:
                mismatched_keys.append(k)

    print("\n📊 --- Checkpoint Load Summary ---")
    print(f"✅ Matched keys: {len(matched_keys)}")
    print(f"⚠️ Mismatched keys: {len(mismatched_keys)}")
    print(f"❌ Missing keys: {len(missing_keys)}")
    print(f"🌀 Unexpected keys (in checkpoint but not in model): {len(unexpected_keys)}")

    print("\n✅ Sample matched layers:")
    print(" ", matched_keys[:10])

    if mismatched_keys:
        print("\n⚠️ Mismatched layer shapes:")
        for k in mismatched_keys[:10]:
            print(f"  {k}: checkpoint {tuple(checkpoint[k].shape)} != model {tuple(model_dict[k].shape)}")

    if missing_keys:
        print("\n❌ Missing layers (not in checkpoint):")
        print(" ", missing_keys[:10])

    if unexpected_keys:
        print("\n🌀 Unexpected layers (in checkpoint, not in model):")
        print(" ", unexpected_keys[:10])

    print("------------------------------------\n")



def load_pretrained_vit_weights(custom_vit, pretrained_weights='/home/s-fx/fun/weights/vit_b_16-c867db91.pth'):
    """
    Maps and loads pretrained ViT-B/16 ImageNet weights into a custom Vision Transformer
    that uses split Q/K/V projection layers and ViT-like naming.

    Works directly for image_size=224, patch_size=16 (no interpolation needed).
    """
    print("🔄 Loading pretrained ViT-B/16 (ImageNet) weights...")
    pretrained_dict = torch.load(pretrained_weights, map_location='cpu')
    custom_state = custom_vit.state_dict()
    mapped_dict = {}

    for k, v in pretrained_dict.items():
        new_k = None

        # ----- Input embeddings -----
        if k == "class_token":
            new_k = "input_layer.cls_token"
        elif k == "conv_proj.weight":
            new_k = "input_layer.patch_embeddings.proj.weight"
        elif k == "conv_proj.bias":
            new_k = "input_layer.patch_embeddings.proj.bias"
        elif k == "encoder.pos_embedding":
            new_k = "input_layer.positional_embeddings"

        # ----- Transformer encoder -----
        elif k.startswith("encoder.layers.encoder_layer_"):
            parts = k.split(".")
            layer_num = parts[2].split("_")[-1]  # e.g. "0" from encoder_layer_0
            suffix = ".".join(parts[3:])

            # --- Normalizations ---
            if "ln_1" in suffix:
                new_k = f"encoder_stack.encoder_stack.{layer_num}.normalisation_stage1"
                if "weight" in suffix:
                    new_k += ".alpha"
                elif "bias" in suffix:
                    new_k += ".bias"

            elif "ln_2" in suffix:
                new_k = f"encoder_stack.encoder_stack.{layer_num}.normalisation_stage2"
                if "weight" in suffix:
                    new_k += ".alpha"
                elif "bias" in suffix:
                    new_k += ".bias"

            # --- Self-Attention (QKV split) ---
            elif "self_attention.in_proj_weight" in suffix:
                qkv_weight = v
                qkv_bias = pretrained_dict[k.replace("in_proj_weight", "in_proj_bias")]

                w_q, w_k, w_v = qkv_weight.chunk(3, dim=0)
                b_q, b_k, b_v = qkv_bias.chunk(3, dim=0)

                mapped_dict[f"encoder_stack.encoder_stack.{layer_num}.mhsa.w_q.weight"] = w_q
                mapped_dict[f"encoder_stack.encoder_stack.{layer_num}.mhsa.w_k.weight"] = w_k
                mapped_dict[f"encoder_stack.encoder_stack.{layer_num}.mhsa.w_v.weight"] = w_v
                mapped_dict[f"encoder_stack.encoder_stack.{layer_num}.mhsa.w_q.bias"] = b_q
                mapped_dict[f"encoder_stack.encoder_stack.{layer_num}.mhsa.w_k.bias"] = b_k
                mapped_dict[f"encoder_stack.encoder_stack.{layer_num}.mhsa.w_v.bias"] = b_v
                continue  # handled fully, skip to next

            elif "self_attention.out_proj.weight" in suffix:
                new_k = f"encoder_stack.encoder_stack.{layer_num}.mhsa.w_out.weight"
            elif "self_attention.out_proj.bias" in suffix:
                new_k = f"encoder_stack.encoder_stack.{layer_num}.mhsa.w_out.bias"

            # --- Feed Forward (MLP) ---
            elif "mlp.fc1.weight" in suffix:
                new_k = f"encoder_stack.encoder_stack.{layer_num}.feed_forward.ff_1.weight"
            elif "mlp.fc1.bias" in suffix:
                new_k = f"encoder_stack.encoder_stack.{layer_num}.feed_forward.ff_1.bias"
            elif "mlp.fc2.weight" in suffix:
                new_k = f"encoder_stack.encoder_stack.{layer_num}.feed_forward.ff_2.weight"
            elif "mlp.fc2.bias" in suffix:
                new_k = f"encoder_stack.encoder_stack.{layer_num}.feed_forward.ff_2.bias"

        # --- Save mapped weight if key matches in your model ---
        if new_k in custom_state and custom_state[new_k].shape == v.shape:
            mapped_dict[new_k] = v

    print(f"✅ Mapped {len(mapped_dict)} weights to your ViT structure.")
    custom_state.update(mapped_dict)
    custom_vit.load_state_dict(custom_state)
    print("🎯 Pretrained ViT weights loaded successfully (for 224×224 images).")

    return custom_vit, custom_state, pretrained_dict


def reinit_classification_head(model):
    linear_layer = model.classification_head[1]  # the final Linear layer
    torch.nn.init.xavier_uniform_(linear_layer.weight)
    if linear_layer.bias is not None:
        torch.nn.init.zeros_(linear_layer.bias)


def compare_state_dicts(model, pretrained_state):
    for name, param in model.named_parameters():
        if name in pretrained_state:
            pretrained_mean = pretrained_state[name].mean().item()
            current_mean = param.data.mean().item()
            print(f"{name}: pretrained mean={pretrained_mean:.4f}, model mean={current_mean:.4f}")


def compare_weight_distributions(pretrained_dict, model, pretrained_key, model_key):
    pretrained_weights = pretrained_dict[pretrained_key].flatten().cpu()
    model_weights = model.state_dict()[model_key].flatten().cpu()

    plt.figure(figsize=(10, 5))
    plt.hist(pretrained_weights.numpy(), bins=80, alpha=0.6, label='Pretrained (ImageNet)', density=True)
    plt.hist(model_weights.numpy(), bins=80, alpha=0.6, label='Your Model', density=True)
    plt.title(f'Weight Distribution Comparison\n{model_key}')
    plt.xlabel('Weight value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig('weight_distr.png')

    # Optional: numeric verification
    #identical = torch.allclose(pretrained_weights, model_weights)
    #print(f'Weights identical: {identical}')
    #print(f'Pretrained mean={pretrained_weights.mean():.6f}, std={pretrained_weights.std():.6f}')
    #print(f'Model mean={model_weights.mean():.6f}, std={model_weights.std():.6f}')

