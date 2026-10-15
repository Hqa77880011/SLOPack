from pathlib import Path
import numpy as np
from slopack.io import write_csv, write_json
from slopack.models import VISION_MODELS


def train_classifier(
    data_path,
    output,
    architecture,
    image_size,
    epochs,
    batch_size,
    lr,
    device,
    seed,
    max_steps,
    pretrained,
    threads,
):
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision.models import get_model

    if architecture not in VISION_MODELS:
        raise ValueError("Training supports the four torchvision classifiers")
    if min(epochs, batch_size, image_size, threads) < 1 or lr <= 0 or max_steps < 0:
        raise ValueError("Invalid training parameters")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    device = torch.device(device)
    with np.load(data_path, allow_pickle=False) as data:
        source = str(data["source"].item())
        train_images = (
            torch.from_numpy(data["images"].copy()).permute(0, 3, 1, 2).float() / 255
        )
        val_images = (
            torch.from_numpy(data["val_images"].copy()).permute(0, 3, 1, 2).float()
            / 255
        )
        train_labels = torch.from_numpy(data["labels"].copy()).long()
        val_labels = torch.from_numpy(data["val_labels"].copy()).long()
    train_loader = DataLoader(
        TensorDataset(train_images, train_labels), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_images, val_labels), batch_size=batch_size
    )
    kwargs = {"image_size": image_size} if architecture == "vit_b_16" else {}
    model = get_model(architecture, weights="DEFAULT" if pretrained else None, **kwargs)
    if architecture == "resnet50":
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
    elif architecture in ("efficientnet_b3", "mobilenet_v3_large"):
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 10)
    else:
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, 10)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def preprocess(images):
        images = functional.interpolate(
            images.to(device),
            (image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        return (images - mean) / std

    history = []
    steps = 0
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        samples = 0
        for images, labels in train_loader:
            if len(labels) == 1:
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(preprocess(images))
            loss = functional.cross_entropy(logits, labels.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced non-finite loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            samples += len(labels)
            steps += 1
            if max_steps and steps >= max_steps:
                break
        if not samples:
            raise ValueError("Training needs at least one batch with two samples")
        model.eval()
        correct = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                correct += (
                    (model(preprocess(images)).argmax(1).cpu() == labels).sum().item()
                )
        history.append(
            {
                "epoch": epoch + 1,
                "steps": steps,
                "train_loss": loss_sum / samples,
                "val_accuracy": correct / len(val_labels),
            }
        )
        print(history[-1], flush=True)
        if max_steps and steps >= max_steps:
            break
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": architecture,
            "num_classes": 10,
            "image_size": image_size,
            "state_dict": {
                key: value.cpu() for key, value in model.state_dict().items()
            },
        },
        output / "model.pt",
    )
    write_csv(output / "history.csv", history)
    write_json(
        output / "training.json",
        {
            "data_source": source,
            "architecture": architecture,
            "image_size": image_size,
            "seed": seed,
            "epochs_completed": len(history),
            "steps": steps,
            "batch_size": batch_size,
            "lr": lr,
            "pretrained": pretrained,
            "normalization": "ImageNet mean/std",
        },
    )
