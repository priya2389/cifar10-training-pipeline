"""Show what each layer of the trained Net does to one CIFAR-10 test image.

Usage:
    python visualize_layers.py                     # picks a confidently-correct test image
    python visualize_layers.py --index 7           # a specific test image (0-9999)
    python visualize_layers.py --weights models/epoch12_model.pt

Saves layers.png (or --out) and opens nothing; view the PNG afterwards.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2

from training import Net

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def run_layers(net, x):
    """Forward pass that keeps every intermediate result."""
    acts = {}
    acts["conv1"] = F.relu(net.conv1(x))
    acts["pool1"] = net.pool(acts["conv1"])
    acts["conv2"] = F.relu(net.conv2(acts["pool1"]))
    acts["pool2"] = net.pool(acts["conv2"])
    acts["flat"] = torch.flatten(acts["pool2"], 1)
    acts["fc1"] = F.relu(net.fc1(acts["flat"]))
    acts["fc2"] = F.relu(net.fc2(acts["fc1"]))
    acts["fc3"] = net.fc3(acts["fc2"])
    acts["probs"] = F.softmax(acts["fc3"], dim=1)
    return acts


def show_maps(fig, gs_row, maps, title, cols):
    """Draw a row (or block) of feature maps."""
    sub = gs_row.subgridspec(int(np.ceil(len(maps) / cols)), cols, wspace=0.05, hspace=0.05)
    for i, m in enumerate(maps):
        ax = fig.add_subplot(sub[i // cols, i % cols])
        ax.imshow(m, cmap="viridis")
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title(title, loc="left", fontsize=11, fontweight="bold", x=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/epoch25_model.pt")
    p.add_argument("--index", type=int, default=None)
    p.add_argument("--out", default="layers.png")
    args = p.parse_args()

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                           transform=transform)

    net = Net()
    net.load_state_dict(torch.load(args.weights))
    net.eval()

    with torch.no_grad():
        idx = args.index
        if idx is None:  # first image the model gets right with >90% confidence
            for i in range(len(testset)):
                img, label = testset[i]
                pr = F.softmax(net(img.unsqueeze(0)), 1)[0]
                if pr.argmax().item() == label and pr.max().item() > 0.9:
                    idx = i
                    break
        img, label = testset[idx]
        acts = run_layers(net, img.unsqueeze(0))

    a = {k: v[0].numpy() for k, v in acts.items()}
    pred = int(a["probs"].argmax())

    fig = plt.figure(figsize=(14, 24))
    gs = fig.add_gridspec(8, 1, height_ratios=[1.0, 1.6, 1.6, 2.4, 2.4, 1.3, 1.3, 1.6], hspace=0.45, top=0.975, bottom=0.02)
    fig.suptitle(f"Inside the CNN — test image #{idx}: true = {CLASSES[label]}, "
                 f"predicted = {CLASSES[pred]} ({a['probs'][pred]:.0%})",
                 fontsize=15, fontweight="bold", y=0.995)

    # 1. input + the 6 learned conv1 filters
    top = gs[0].subgridspec(1, 7, wspace=0.15)
    ax = fig.add_subplot(top[0, 0])
    ax.imshow(np.transpose(img.numpy() / 2 + 0.5, (1, 2, 0)))
    ax.set_title("Input 3×32×32", fontsize=11, fontweight="bold")
    ax.axis("off")
    w = net.conv1.weight.detach().numpy()  # 6×3×5×5
    for i in range(6):
        f = np.transpose(w[i], (1, 2, 0))
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        ax = fig.add_subplot(top[0, i + 1])
        ax.imshow(f, interpolation="nearest")
        ax.set_title(f"conv1 filter {i}", fontsize=9)
        ax.axis("off")

    show_maps(fig, gs[1], a["conv1"], "conv1 + ReLU → 6×28×28  (each filter's response; bright = pattern found)", 6)
    show_maps(fig, gs[2], a["pool1"], "maxpool → 6×14×14  (keep the strongest value in each 2×2 block)", 6)
    show_maps(fig, gs[3], a["conv2"], "conv2 + ReLU → 16×10×10  (combinations of conv1 patterns)", 8)
    show_maps(fig, gs[4], a["pool2"], "maxpool → 16×5×5  (flattened into 400 numbers)", 8)

    for row, key, title in [(5, "fc1", "fc1 + ReLU → 120 features"),
                            (6, "fc2", "fc2 + ReLU → 84 features")]:
        ax = fig.add_subplot(gs[row])
        ax.bar(range(len(a[key])), a[key], color="#4C72B0", width=0.8)
        ax.set_title(title + f"  ({(a[key] > 0).sum()} active, rest zeroed by ReLU)",
                     loc="left", fontsize=11, fontweight="bold")
        ax.set_xlim(-1, len(a[key])); ax.set_ylabel("activation")

    ax = fig.add_subplot(gs[7])
    colors = ["#DD8452" if i == pred else "#BBBBBB" for i in range(10)]
    ax.bar(CLASSES, a["probs"], color=colors)
    for i, pr in enumerate(a["probs"]):
        ax.text(i, pr + 0.01, f"{pr:.0%}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel("probability")
    ax.set_title("fc3 → 10 scores → softmax = class probabilities", loc="left",
                 fontsize=11, fontweight="bold")

    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"image #{idx}  true={CLASSES[label]}  pred={CLASSES[pred]}  saved {args.out}")


if __name__ == "__main__":
    main()
