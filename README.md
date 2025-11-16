# ViTransformer Classificator

Trained Vision Transformer on the Retinopathy Dataset [Dataset](https://www.kaggle.com/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy). The model is pretrained on Imagenet and then finetuned. 

**Class Distribution** (class imbalance)

![Class Distribution](./assets/class_distribution.png)

**Loss**

![Loss Function](./assets/loss_epoch.png)

**Weight Distribution**

![Weight Distribution](./assets/vit_layer_comparison.gif)


## Pre-Quantisation

**Metric**

![Metrics](./assets/metrics.png)

**Confusion Matrix**

![Confusion Matrix](./assets/confusion_matrix.png)


## After-Quantisation

TODO

## Testing

- High Accuracy ViT Base Model
- Mid Accuracy ViT Base Model
- Overfitted ViT Base Model
- Check if quantisation has regularisation effect on overfitted Model
- Check how accuracy drops compared to high/mid accuracy models
- UMAP feature space base model vs quantised model comparision

## TODO

- [x] Evaluation Script
- [x] Inference Script
- [x] Plot Loss
- [x] Validation Loop
- [ ] Quantisation 
- [ ] Test Quantisation
- [x] Refactor Code
- [x] Save model parameters during training in txt file
- [ ] Measure model size, speedup, ...
- [ ] Fix class imbalance, black borders in images
