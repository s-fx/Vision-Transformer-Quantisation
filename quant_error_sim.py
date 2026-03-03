import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from src.dino_vit import load_dino, visualise_features, get_transforms
from src.dataset.dataset import RetinopathyFullDataset, CIFAR10Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class DinoQuantizationEvaluator:

    def __init__(self, model, device="cuda", bits=8):
        self.model = model.to(device)
        self.device = device
        self.bits = bits
        self.handles = []
        self.activations = {}
        self.stats = defaultdict(list)

    # -------------------------------------------------
    # Fake Quantisierung (symmetrisch)
    # -------------------------------------------------
    def fake_quant(self, x):
        qmin = -2**(self.bits - 1)
        qmax = 2**(self.bits - 1) - 1

        scale = x.abs().max() / qmax + 1e-8
        x_q = torch.clamp((x / scale).round(), qmin, qmax)
        return x_q * scale

    # -------------------------------------------------
    # Hook Registrierung (architektur-spezifisch)
    # -------------------------------------------------
    def register_hooks(self):

        def hook(name):
            def fn(module, inp, out):
                self.activations[name] = out.detach()
            return fn

        # Transformer Blocks
        for i, block in enumerate(self.model.transformer.blocks):

            self.handles.append(
                block.norm1.register_forward_hook(hook(f"block{i}_norm1"))
            )

            self.handles.append(
                block.norm2.register_forward_hook(hook(f"block{i}_norm2"))
            )

            self.handles.append(
                block.attn.qkv.register_forward_hook(hook(f"block{i}_qkv"))
            )

            self.handles.append(
                block.attn.proj.register_forward_hook(hook(f"block{i}_attn_proj"))
            )

            self.handles.append(
                block.mlp.fc1.register_forward_hook(hook(f"block{i}_fc1"))
            )

            self.handles.append(
                block.mlp.fc2.register_forward_hook(hook(f"block{i}_fc2"))
            )

        # Final LayerNorm
        self.handles.append(
            self.model.transformer.norm.register_forward_hook(
                hook("final_norm")
            )
        )

        # Classifier
        self.handles.append(
            self.model.classifier[0].register_forward_hook(
                hook("classifier_fc1")
            )
        )

        self.handles.append(
            self.model.classifier[3].register_forward_hook(
                hook("classifier_fc2_logits")
            )
        )

    def remove_hooks(self):
        for h in self.handles:
            h.remove()

    # -------------------------------------------------
    # Metriken
    # -------------------------------------------------
    def compute_metrics(self, x, x_q, name):

        mse = F.mse_loss(x_q, x).item()
        mae = torch.mean(torch.abs(x_q - x)).item()
        rel = torch.mean(torch.abs(x_q - x) / (torch.abs(x) + 1e-8)).item()

        signal_power = torch.mean(x**2)
        noise_power = torch.mean((x - x_q)**2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-12))

        dynamic_range = (x.max() - x.min()).item()

        self.stats[name].append({
            "MSE": mse,
            "MAE": mae,
            "RelativeError": rel,
            "SNR_dB": snr.item(),
            "Range": dynamic_range
        })

    # -------------------------------------------------
    # Softmax Sensitivität
    # -------------------------------------------------
    def compute_softmax_kl(self, logits):

        logits_q = self.fake_quant(logits)

        p = F.softmax(logits, dim=-1)
        p_q = F.softmax(logits_q, dim=-1)

        kl = F.kl_div(p_q.log(), p, reduction='batchmean').item()

        self.stats["softmax_KL"].append({"KL": kl})

    # -------------------------------------------------
    # Evaluation pro Batch
    # -------------------------------------------------
    def evaluate_batch(self, x):

        self.activations = {}

        with torch.no_grad():
            logits = self.model(x.to(self.device))

        # Layer-wise Analyse
        for name, act in self.activations.items():
            act_q = self.fake_quant(act)
            self.compute_metrics(act, act_q, name)

        # Softmax Sensitivität (Classifier Output)
        self.compute_softmax_kl(logits)

    # -------------------------------------------------
    # Aggregation
    # -------------------------------------------------
    def summarize(self):

        summary = {}

        for layer, entries in self.stats.items():
            summary[layer] = {}

            keys = entries[0].keys()
            for k in keys:
                vals = [e[k] for e in entries]
                summary[layer][k] = np.mean(vals)

        return summary

class QuantBlockWrapper(nn.Module):

    def __init__(self, block, bits=8):
        super().__init__()
        self.block = block
        self.bits = bits

    def fake_quant(self, x):
        qmin = -2**(self.bits - 1)
        qmax = 2**(self.bits - 1) - 1
        scale = x.abs().max() / qmax + 1e-8
        x_q = torch.clamp((x / scale).round(), qmin, qmax)
        return x_q * scale

    def fake_quant_asymmetric(self, x):

        qmin = 0
        qmax = 2**self.bits - 1

        x_min = x.min()
        x_max = x.max()

        scale = (x_max - x_min) / (qmax - qmin + 1e-8)

        zero_point = qmin - torch.round(x_min / (scale + 1e-8))
        zero_point = torch.clamp(zero_point, qmin, qmax)

        q = torch.round(x / scale + zero_point)
        q = torch.clamp(q, qmin, qmax)

        x_q = scale * (q - zero_point)

        return x_q

    def forward(self, x):

        # Quantisiere Block Input
        x = self.fake_quant_asymmetric(x)

        out = self.block(x)

        # Quantisiere Block Output
        out = self.fake_quant(out)

        return out


class BlockSensitivityAnalyzer:

    def __init__(self, model, device="cuda", bits=8):
        self.model = model.to(device)
        self.device = device
        self.bits = bits
        self.n_blocks = len(model.transformer.blocks)

    def measure_block_sensitivity(self, dataloader):

        sensitivities = []

        # Referenz Forward (FP32)
        ref_outputs = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                logits = self.model(x)
                ref_outputs.append(logits.cpu())

        ref_outputs = torch.cat(ref_outputs)

        # Für jeden Block separat quantisieren
        for i in range(self.n_blocks):

            print(f"Analyzing Block {i}")

            original_block = self.model.transformer.blocks[i]

            # Block ersetzen
            self.model.transformer.blocks[i] = QuantBlockWrapper(
                original_block, bits=self.bits
            )

            q_outputs = []
            with torch.no_grad():
                for x, _ in dataloader:
                    x = x.to(self.device)
                    logits = self.model(x)
                    q_outputs.append(logits.cpu())

            q_outputs = torch.cat(q_outputs)

            # MSE zwischen FP32 und quantisiert
            mse = torch.mean((ref_outputs - q_outputs)**2).item()
            sensitivities.append(mse)

            # Block zurücksetzen
            self.model.transformer.blocks[i] = original_block

        return sensitivities


class QuantizationPlotter:

    def __init__(self, results):
        self.results = results
        self.grouped = self._group_layers()

    # -------------------------------------------------
    # Layer automatisch klassifizieren
    # -------------------------------------------------
    def _classify_layer(self, name):

        if "norm" in name:
            return "LayerNorm"

        elif "qkv" in name or "attn_proj" in name:
            return "Attention"

        elif "fc1" in name or "fc2" in name:
            return "MLP"

        elif "classifier" in name:
            return "Classifier"

        else:
            return "Other"

    # -------------------------------------------------
    # Nach Block + Typ gruppieren
    # -------------------------------------------------
    def _group_layers(self):

        grouped = defaultdict(lambda: defaultdict(dict))

        for name, metrics in self.results.items():

            # Block index extrahieren
            match = re.search(r'block(\d+)', name)
            block_id = int(match.group(1)) if match else -1

            layer_type = self._classify_layer(name)

            grouped[layer_type][block_id] = metrics

        return grouped

    # -------------------------------------------------
    # Plot pro Layer-Typ über Blocks
    # -------------------------------------------------
    def plot_metric_per_block(self, metric="MSE", figname=''):

        plt.figure(figsize=(10,6))

        for layer_type, blocks in self.grouped.items():

            block_ids = sorted(b for b in blocks.keys() if b >= 0)

            values = [blocks[b][metric] for b in block_ids]

            plt.plot(block_ids, values, marker='o', label=layer_type)

        plt.xlabel("Transformer Block Index")
        plt.ylabel(metric)
        plt.title(f"{metric} per Block grouped by Layer Type")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(figname)
        plt.show()

    # -------------------------------------------------
    # Aggregierter Mittelwert pro Layer-Typ
    # -------------------------------------------------
    def plot_aggregate_per_type(self, metric="MSE", figname=''):

        types = []
        values = []

        for layer_type, blocks in self.grouped.items():

            vals = []
            for b, metrics in blocks.items():
                if b >= 0:
                    vals.append(metrics[metric])

            if len(vals) > 0:
                types.append(layer_type)
                values.append(np.mean(vals))

        plt.figure(figsize=(6,5))
        plt.bar(types, values)
        plt.ylabel(metric)
        plt.title(f"Average {metric} per Layer Type")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(figname)
        plt.show()

    # -------------------------------------------------
    # Heatmap (Block × Layer-Type)
    # -------------------------------------------------
    def plot_heatmap(self, metric="MSE", figname=''):

        layer_types = list(self.grouped.keys())
        max_block = max(
            b for blocks in self.grouped.values()
            for b in blocks.keys()
            if b >= 0
        )

        matrix = np.zeros((len(layer_types), max_block+1))

        for i, layer_type in enumerate(layer_types):
            for b, metrics in self.grouped[layer_type].items():
                if b >= 0:
                    matrix[i, b] = metrics[metric]

        plt.figure(figsize=(10,6))
        plt.imshow(matrix, aspect='auto')
        plt.colorbar(label=metric)
        plt.yticks(range(len(layer_types)), layer_types)
        plt.xlabel("Block Index")
        plt.title(f"{metric} Heatmap")
        plt.tight_layout()
        plt.savefig(figname)
        plt.show()

if __name__ == '__main__':

    root = './runs/run_dino_cifar_backbone'
    data_root = '/home/s-fx/fun/datasets/CIFAR-10-dataset'
    params_path = os.path.join(root, 'params.json')
    ckpt_path = os.path.join(root, 'epoch_2_model.pth')
    loss_dict = os.path.join(root, 'loss_dict.pkl')
    device = 'cuda'
    #images_path = glob.glob(f'{data_root}/single_example/*/*.jpg')
    _, val_transform = get_transforms()

    dataset = CIFAR10Dataset(data_root, val_transform, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


    # ------------ BASE MODEL --------------
    model = load_dino(params_path, ckpt_path, device)
    model.eval()
    model.to(device)


    # misst lokale Layer-Sensitivität.=
    evaluator = DinoQuantizationEvaluator(model, device="cuda", bits=8)
    evaluator.register_hooks()

    for images, _ in dataloader:
        evaluator.evaluate_batch(images)

    results = evaluator.summarize()
    print(results)


    # Misst den globalen Einfluss eines Blocks auf den Final Output
    # Welche Bloecke sind kritisch
    analyzer = BlockSensitivityAnalyzer(model, device="cuda", bits=8)
    sens = analyzer.measure_block_sensitivity(dataloader)

    plt.figure(figsize=(10,5))
    plt.plot(range(len(sens)), sens, marker='o')
    plt.xlabel("Transformer Block Index")
    plt.ylabel("Output MSE (FP32 vs Quantized)")
    plt.title("Block Sensitivity to Quantization")
    plt.grid(True)
    plt.show()

    plotter = QuantizationPlotter(results)

    plotter.plot_metric_per_block(metric="MSE", figname='mse_asym')
    plotter.plot_metric_per_block(metric="SNR_dB", figname='snr_dB_asym')

    plotter.plot_aggregate_per_type(metric="MSE", figname='aggregate_per_type_mse_asym')
    plotter.plot_heatmap(metric="MSE", figname='heatmap_mse_asym')

