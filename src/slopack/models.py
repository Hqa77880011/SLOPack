import os
from pathlib import Path
import numpy as np

VISION_MODELS = ("resnet50", "efficientnet_b3", "mobilenet_v3_large", "vit_b_16")


def load_model(spec, device, precision="fp32"):
    import torch
    from torchvision.models import get_model

    torch.manual_seed(spec.get("model_seed", 0))
    architecture = spec["architecture"]
    weights = spec.get("weights", "random")
    checkpoint = None
    if spec.get("checkpoint"):
        checkpoint = torch.load(
            spec["checkpoint"], map_location="cpu", weights_only=True
        )
        if checkpoint["architecture"] != architecture:
            raise ValueError("Checkpoint architecture does not match model config")
        weights = "random"
    if architecture in VISION_MODELS:
        kwargs = {}
        if architecture == "vit_b_16":
            if len(set(spec["shapes"])) != 1:
                raise ValueError("ViT requires one fixed image size per replica")
            kwargs["image_size"] = spec["shapes"][0]
        if checkpoint:
            kwargs["num_classes"] = checkpoint["num_classes"]
        model = get_model(
            architecture, weights=None if weights == "random" else weights, **kwargs
        )
    elif architecture == "bert_base":
        from transformers import BertConfig, BertModel

        model = (
            BertModel(BertConfig())
            if weights == "random"
            else BertModel.from_pretrained(weights)
        )
    elif architecture == "yolov5m":
        visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            model = torch.hub.load(
                "ultralytics/yolov5:f9578796b8a64889aa9e66dc06914d19f40d338e",
                "yolov5m",
                pretrained=weights != "random",
                autoshape=False,
                trust_repo=True,
                device="cpu",
                _verbose=False,
            )
        finally:
            if visibility is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = visibility
        if getattr(model, "pt", False):
            model = model.model
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    if checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(device)
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 execution requires a CUDA device")
    model = model.to(device).eval()
    if precision == "fp16":
        model = model.half()
    return model


def load_payloads(path=None, count=64, seed=0):
    import torch

    if path:
        with np.load(Path(path), allow_pickle=False) as data:
            images = data["val_images"] if "val_images" in data else data["images"]
            images = images.copy()
            tokens = data["tokens"].copy() if "tokens" in data else None
    else:
        rng = np.random.default_rng(seed)
        images = rng.integers(0, 256, (count, 32, 32, 3), dtype=np.uint8)
        tokens = None
    if images.ndim != 4 or images.shape[-1] != 3 or len(images) == 0:
        raise ValueError("Payload images must be nonempty NHWC RGB arrays")
    bank = {"images": torch.from_numpy(np.ascontiguousarray(images)).share_memory_()}
    if tokens is not None:
        if (
            tokens.ndim != 2
            or len(tokens) == 0
            or tokens.min() < 0
            or (tokens.max() >= 30522)
        ):
            raise ValueError("BERT tokens must be a nonempty N x S array in [0, 30522)")
        bank["tokens"] = torch.from_numpy(tokens.astype(np.int64)).share_memory_()
    return bank


def make_inputs(spec, shape, payload_ids, bank, device, precision):
    import torch
    import torch.nn.functional as functional

    if spec["architecture"] == "bert_base":
        if "tokens" in bank:
            tokens = bank["tokens"][torch.as_tensor(payload_ids) % len(bank["tokens"])]
            if tokens.shape[1] < shape:
                tokens = functional.pad(tokens, (0, shape - tokens.shape[1]), value=0)
            ids = tokens[:, :shape].to(device)
            mask = ids.ne(0).long()
        else:
            indices = torch.as_tensor(payload_ids, dtype=torch.long)[:, None]
            ids = (indices * 101 + torch.arange(shape)[None, :] * 13 + 100) % 30522
            ids[:, 0] = 101
            ids[:, -1] = 102
            ids = ids.to(device)
            mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}
    indices = torch.as_tensor(payload_ids) % len(bank["images"])
    images = bank["images"][indices].permute(0, 3, 1, 2).float().div_(255)
    images = functional.interpolate(
        images, size=(shape, shape), mode="bilinear", align_corners=False
    )
    if spec["architecture"] != "yolov5m":
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = (images - mean) / std
    dtype = torch.float16 if precision == "fp16" else torch.float32
    return images.to(device=device, dtype=dtype)


def forward(model, inputs):
    return model(**inputs) if isinstance(inputs, dict) else model(inputs)


def prediction_summary(spec, output):
    if spec["architecture"] == "bert_base":
        return output.pooler_output.float().norm(dim=1).detach().cpu().tolist()
    if spec["architecture"] == "yolov5m":
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        return tensor[..., 4].float().amax(dim=1).detach().cpu().tolist()
    return output.argmax(dim=1).detach().cpu().tolist()


def parameter_mb(model):
    return (
        sum(
            (
                t.numel() * t.element_size()
                for t in list(model.parameters()) + list(model.buffers())
            )
        )
        / 2**20
    )
