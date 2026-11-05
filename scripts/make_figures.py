"""Regenerate the README figures that come from the trained models.

dataset_examples.jpg and rf_detr_preliminary.jpg are not generated here: they
come from notebook outputs and the project report.

Needs the git-ignored local files: the DINOv2 checkpoints in runs/, the cached
segmentation tensors in data/cache_seg/, and the images in data/ and
patches_dataset/. ffmpeg is required for the GIF.

    python scripts/make_figures.py                 # DINOv2 code from torch.hub
    python scripts/make_figures.py --dinov2-src DIR  # local dinov2 checkout
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import font_manager
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from train_dinov2_segmentation import DinoV2Segmentation  # noqa: E402

ASSETS = ROOT / "docs" / "assets"
BOXES_CSV = ROOT / "data" / "TreeBoxes_v0.11" / "random.csv"
IMAGES = ROOT / "data" / "TreeBoxes_v0.11" / "images"
CLASSIFIER = ROOT / "runs" / "dinov2_vits14_normal" / "best_model_dinov2_vits14_binary_head_0.001_10.pth"
SEGMENTER = ROOT / "runs" / "dinov2_seg1" / "model_best.pth"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

DEMO_IMAGE = "tile_75_37_lat_50.86578_lon_4.71026.png"
SECOND_IMAGE = "tile_58_87_lat_50.89147_lon_4.69642.png"

# Same preprocessing as training (PatchDataset): resize + ToTensor, no ImageNet normalisation.
TO_TENSOR = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# Not stored in runs/: read from the outputs of notebooks/02_classification/06_clip.ipynb.
CLIP_LEARNED_PROMPTS_AUROC = 0.9236


class DinoV2BinaryClassifier(nn.Module):
    def __init__(self, backbone, feat_dim=384, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def load_backbone(src):
    # pretrained=False: the checkpoints below already contain the backbone weights.
    if src:
        return torch.hub.load(src, "dinov2_vits14", source="local", pretrained=False)
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=False)


def load_weights(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval().to(DEVICE)


def scan(model, image, grid):
    """Run the classifier over a nominal grid with 50% overlap, as in notebook 13."""
    patch = image.width // grid
    positions = np.rint(np.linspace(0, image.width - patch, 2 * grid - 1)).astype(int)
    probs = np.zeros((len(positions), len(positions)), dtype=np.float32)
    with torch.no_grad():
        for row, y in enumerate(positions):
            crops = [image.crop((int(x), int(y), int(x) + patch, int(y) + patch)) for x in positions]
            logits = model(torch.stack([TO_TENSOR(c) for c in crops]).to(DEVICE))
            probs[row] = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return positions, patch, probs


def density_map(size, positions, patch, probs, upto=None):
    """Max aggregation: each pixel keeps the highest probability of the windows covering it."""
    density = np.full((size, size), -1.0, dtype=np.float32)  # -1 marks pixels not scanned yet
    for i, (row, col) in enumerate(np.ndindex(probs.shape)):
        if upto is not None and i >= upto:
            break
        y, x = positions[row], positions[col]
        window = density[y:y + patch, x:x + patch]
        np.maximum(window, probs[row, col], out=window)
    return density


def boxes_for(filename):
    boxes = pd.read_csv(BOXES_CSV, low_memory=False)
    return boxes.loc[boxes["filename"] == filename, ["xmin", "ymin", "xmax", "ymax"]].to_numpy()


def overlay(rgb, density, alpha=0.62):
    """Viridis over a dimmed greyscale image; unscanned pixels stay dimmed."""
    grey = np.asarray(Image.fromarray(rgb).convert("L").convert("RGB"), dtype=np.float32) * 0.45
    heat = plt.get_cmap("viridis")(np.clip(density, 0, 1))[..., :3] * 255
    scanned = (density >= 0)[..., None]
    return np.where(scanned, grey * (1 - alpha) + heat * alpha, grey).astype(np.uint8)


def sliding_window_gif(model):
    image = Image.open(IMAGES / DEMO_IMAGE).convert("RGB")
    positions, patch, probs = scan(model, image, grid=16)
    rgb = np.asarray(image)
    size, panel, top, gap, step = image.width, 360, 44, 20, 12
    scale = panel / size
    width, height = panel * 2 + gap * 3, panel + top + 46
    font_path = font_manager.findfont("DejaVu Sans")
    title, small = ImageFont.truetype(font_path, 16), ImageFont.truetype(font_path, 13)
    ink, muted, yellow = (240, 239, 236), (170, 170, 165), (255, 214, 10)

    def frame(done, show_boxes=False):
        canvas = Image.new("RGB", (width, height), (17, 17, 17))
        draw = ImageDraw.Draw(canvas)
        left, right = gap, gap * 2 + panel
        canvas.paste(image.resize((panel, panel)), (left, top))
        density = density_map(size, positions, patch, probs, upto=done)
        canvas.paste(Image.fromarray(overlay(rgb, density)).resize((panel, panel)), (right, top))
        draw.text((left, 13), "Sliding window over the image", font=title, fill=ink)
        draw.text((right, 13), "Tree probability map", font=title, fill=ink)
        if show_boxes:
            for x1, y1, x2, y2 in boxes_for(DEMO_IMAGE) * scale:
                draw.rectangle((right + x1, top + y1, right + x2, top + y2), outline=(255, 77, 77), width=2)
            footer = f"All {probs.size} windows done. Red boxes: ground-truth tree crowns"
        else:
            row, col = np.unravel_index(done - 1, probs.shape)
            x, y = left + positions[col] * scale, top + positions[row] * scale
            draw.rectangle((x, y, x + patch * scale, y + patch * scale), outline=yellow, width=3)
            footer = f"Window {done} / {probs.size}    DINOv2 P(tree) = {probs[row, col]:.2f}"
        draw.text((left, top + panel + 14), footer, font=small, fill=muted)
        return canvas

    with tempfile.TemporaryDirectory() as tmp:
        steps = [*range(step, probs.size, step), probs.size]
        frames = [frame(done) for done in steps] + [frame(probs.size, show_boxes=True)] * 32
        for i, im in enumerate(frames):
            im.save(f"{tmp}/{i:04d}.png")
        graph = "split[a][b];[a]palettegen=stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12", "-i", f"{tmp}/%04d.png",
             "-lavfi", graph, "-loop", "0", str(ASSETS / "sliding_window.gif")],
            check=True,
        )


def density_grids(model):
    grids = (8, 16, 32)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2))
    for axes_row, filename in zip(axes, (DEMO_IMAGE, SECOND_IMAGE)):
        image = Image.open(IMAGES / filename).convert("RGB")
        for ax, grid in zip(axes_row, grids):
            positions, patch, probs = scan(model, image, grid)
            ax.imshow(overlay(np.asarray(image), density_map(image.width, positions, patch, probs), alpha=0.55))
            for x1, y1, x2, y2 in boxes_for(filename):
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#ff4d4d", linewidth=0.9))
            ax.set_xticks([]), ax.set_yticks([])
            if ax in axes[0]:
                ax.set_title(f"{grid}×{grid} grid, {probs.size} windows", fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.03, hspace=0.03)
    fig.savefig(ASSETS / "density_grids.jpg", dpi=130, pil_kwargs={"quality": 88})
    plt.close(fig)


def segmentation_figure(model):
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    samples = []
    for path in sorted((ROOT / "data" / "cache_seg").glob("test_*.pt")):
        item = torch.load(path, weights_only=True)
        with torch.no_grad():
            prob = torch.softmax(model(item["img"].float()[None].to(DEVICE)), dim=1)[0, 1].cpu().numpy()
        truth, pred = item["mask"].numpy() == 1, prob >= 0.5
        iou = (pred & truth).sum() / (pred | truth).sum()
        rgb = np.clip(item["img"].float().permute(1, 2, 0).numpy() * std + mean, 0, 1)
        samples.append((iou, rgb, truth, pred, prob))
    ious = [s[0] for s in samples]
    print(f"segmentation: {len(samples)} test images, mean IoU {np.mean(ious):.3f}")

    samples.sort(key=lambda s: -s[0])
    chosen = (samples[0], samples[len(samples) // 2], samples[-1])  # best, median, worst
    colours = {"tp": "#1baf7a", "fp": "#eb6834", "fn": "#2a78d6"}
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.6))
    for row, (iou, rgb, truth, pred, prob) in zip(axes, chosen):
        grey = rgb.mean(axis=2, keepdims=True).repeat(3, axis=2) * 0.55
        errors = grey.copy()
        for name, region in (("tp", pred & truth), ("fp", pred & ~truth), ("fn", ~pred & truth)):
            errors[region] = 0.35 * grey[region] + 0.65 * np.array(matplotlib.colors.to_rgb(colours[name]))
        row[0].imshow(rgb)
        row[1].imshow(truth, cmap="gray")
        row[2].imshow(prob, cmap="viridis", vmin=0, vmax=1)
        row[3].imshow(errors)
        row[0].set_ylabel(f"IoU {iou:.2f}", fontsize=13)
        for ax in row:
            ax.set_xticks([]), ax.set_yticks([])
    for ax, name in zip(axes[0], ("Aerial image", "Ground truth", "Predicted P(tree)", "Errors")):
        ax.set_title(name, fontsize=13)
    fig.legend(
        handles=[Patch(color=colours["tp"], label="Correct tree pixels"),
                 Patch(color=colours["fp"], label="False positives"),
                 Patch(color=colours["fn"], label="Missed tree pixels")],
        loc="lower center", ncol=3, frameon=False, fontsize=12,
    )
    fig.subplots_adjust(left=0.05, right=0.995, top=0.96, bottom=0.06, wspace=0.03, hspace=0.04)
    fig.savefig(ASSETS / "segmentation_test.jpg", dpi=125, pil_kwargs={"quality": 88})
    plt.close(fig)


def patches_figure():
    root = ROOT / "patches_dataset" / "patches_v3_seed42_pad1.6_iouth0.09"
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(2, 7, figsize=(11, 3.4))
    for row, label in zip(axes, ("positive", "negative")):
        files = sorted((root / label).glob("*.png"))
        for ax, index in zip(row, rng.choice(len(files), len(row), replace=False)):
            ax.imshow(Image.open(files[index]))
            ax.set_xticks([]), ax.set_yticks([])
        row[0].set_ylabel("Tree" if label == "positive" else "No tree", fontsize=12)
    fig.subplots_adjust(left=0.035, right=0.998, top=0.995, bottom=0.005, wspace=0.03, hspace=0.03)
    fig.savefig(ASSETS / "patches.jpg", dpi=120, pil_kwargs={"quality": 88})
    plt.close(fig)


def auroc_from_run(name):
    return json.loads((ROOT / "runs" / name / "results.json").read_text())["test"]["auroc"]


def rounded_bar(ax, value, y, height, colour, radius_px=4):
    """Bar with a rounded data end and a square end on the baseline."""
    box = ax.get_window_extent()
    rx = radius_px * np.diff(ax.get_xlim())[0] / box.width
    ry = radius_px * abs(np.diff(ax.get_ylim())[0]) / box.height
    y0, y1 = y - height / 2, y + height / 2
    vertices = [(0, y0), (value - rx, y0), (value, y0), (value, y0 + ry), (value, y1 - ry),
                (value, y1), (value - rx, y1), (0, y1), (0, y0)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
             MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=colour, edgecolor="none"))


def results_chart():
    rows = [
        ("DINOv2 ViT-S/14, frozen + MLP head", "50k trainable", auroc_from_run("dinov2_vits14_normal")),
        ("ViT from scratch", "11M trainable", auroc_from_run("vit_v2_seed42")),
        ("CLIP + learned prompts", "5k trainable", CLIP_LEARNED_PROMPTS_AUROC),
        ("CNN from scratch", "24k trainable", auroc_from_run("cnn_boxes_v3_normal")),
        ("ResNet-18, frozen + linear probe", "1k trainable", auroc_from_run("resnet18_ft_seed42")),
    ]
    themes = {
        "light": dict(ink="#0b0b0b", soft="#52514e", grid="#e4e3df", bar="#2a78d6"),
        "dark": dict(ink="#f0efec", soft="#c3c2b7", grid="#30363d", bar="#3987e5"),
    }
    for name, theme in themes.items():
        fig, ax = plt.subplots(figsize=(9.2, 3.8))
        fig.subplots_adjust(left=0.335, right=0.97, top=0.86, bottom=0.1)
        ax.set_xlim(0, 1.07)
        ax.set_ylim(len(rows) - 0.45, -0.55)
        ax.set_xticks(np.arange(0, 1.01, 0.25))
        ax.tick_params(axis="x", colors=theme["soft"], length=0, labelsize=10)
        ax.set_yticks([])
        ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.axvline(0.5, color=theme["soft"], linewidth=1, linestyle=(0, (4, 3)))
        ax.text(0.505, len(rows) - 0.52, "chance", color=theme["soft"], fontsize=9.5, va="bottom")
        for y, (label, params, value) in enumerate(rows):
            rounded_bar(ax, value, y, 0.46, theme["bar"])
            ax.text(value + 0.012, y, f"{value:.3f}", va="center", color=theme["ink"], fontsize=11, fontweight="bold")
            ax.text(-0.015, y - 0.13, label, ha="right", va="center", color=theme["ink"], fontsize=11)
            ax.text(-0.015, y + 0.2, params, ha="right", va="center", color=theme["soft"], fontsize=9.5)
        fig.text(0.015, 0.93, "Test AUROC on 3,965 held-out patches", color=theme["ink"], fontsize=14, fontweight="bold")
        fig.savefig(ASSETS / f"results_auroc_{name}.png", dpi=150, transparent=True)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dinov2-src", help="local checkout of facebookresearch/dinov2")
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)

    results_chart()
    patches_figure()
    backbone = load_backbone(args.dinov2_src)
    classifier = load_weights(DinoV2BinaryClassifier(backbone), CLASSIFIER)
    sliding_window_gif(classifier)
    density_grids(classifier)
    segmentation_figure(load_weights(DinoV2Segmentation(backbone), SEGMENTER))


if __name__ == "__main__":
    main()
