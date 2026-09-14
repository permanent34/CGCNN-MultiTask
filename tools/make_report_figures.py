#!/usr/bin/env python
"""Generate report diagrams and a real-metrics chart."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
BLUE = "#2f6fbb"
LIGHT_BLUE = "#dce9f7"
ORANGE = "#e58b32"
LIGHT_ORANGE = "#fbe3cb"
GREEN = "#3c8b68"
LIGHT_GREEN = "#dcefe6"
GRAY = "#5f6673"


def box(ax, x, y, w, h, text, facecolor=LIGHT_BLUE, edgecolor=BLUE, fontsize=11):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.6,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end, color=GRAY, style="-|>", mutation_scale=16):
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=1.6,
        color=color,
    ))


def clean(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def cgcnn_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.2))
    clean(ax)
    labels = [
        "Crystal\nstructure",
        "CIF + labels",
        "Crystal graph\nnodes + edges",
        "CGCNN\n3 graph convolutions",
        "Mean pooling\ncrystal feature",
        "Prediction\nproperty/properties",
    ]
    width, height, gap = 0.145, 0.34, 0.027
    for i, label in enumerate(labels):
        x = 0.035 + i * (width + gap)
        box(ax, x, 0.33, width, height, label)
        if i < len(labels) - 1:
            arrow(ax, (x + width + 0.003, 0.5), (x + width + gap - 0.003, 0.5))
    ax.text(0.5, 0.88, "CGCNN: Crystal Structure to Material Property", ha="center", fontsize=16, weight="bold")
    ax.text(0.5, 0.12, "Local atomic environments are encoded as a graph and aggregated into one crystal representation.", ha="center", fontsize=10, color=GRAY)
    save(fig, "cgcnn_pipeline.png")


def crystal_graph():
    fig, ax = plt.subplots(figsize=(8, 6))
    clean(ax)
    nodes = {
        "i": (0.48, 0.50, "center atom i"),
        "a": (0.23, 0.73, "neighbor j"),
        "b": (0.76, 0.72, "neighbor k"),
        "c": (0.78, 0.28, "neighbor l"),
        "d": (0.22, 0.27, "neighbor m"),
        "e": (0.50, 0.86, "periodic image"),
    }
    edges = [("i", "a"), ("i", "b"), ("i", "c"), ("i", "d"), ("a", "e"), ("b", "e")]
    for start, end in edges:
        x1, y1, _ = nodes[start]
        x2, y2, _ = nodes[end]
        ax.plot([x1, x2], [y1, y2], color="#9ba9bc", linewidth=2, zorder=1)
    for key, (x, y, label) in nodes.items():
        color = ORANGE if key == "i" else BLUE
        ax.add_patch(Circle((x, y), 0.065, facecolor=color, edgecolor="white", linewidth=2, zorder=2))
        ax.text(x, y, key, color="white", ha="center", va="center", fontsize=13, weight="bold", zorder=3)
        ax.text(x, y - 0.105, label, ha="center", va="top", fontsize=9)
    x1, y1, _ = nodes["i"]
    x2, y2, _ = nodes["a"]
    ax.text((x1 + x2) / 2 - 0.02, (y1 + y2) / 2 + 0.03, "edge feature u$_{ij}$", rotation=32, fontsize=11, color=GRAY)
    ax.text(0.5, 0.95, "Crystal graph: atoms as nodes, neighbors as edges", ha="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.08, "Node: 92-d atom feature    Edge: 41-d Gaussian distance feature", ha="center", fontsize=10, color=GRAY)
    save(fig, "crystal_graph.png")


def multitask_model():
    fig, ax = plt.subplots(figsize=(11, 6))
    clean(ax)
    box(ax, 0.04, 0.43, 0.18, 0.18, "Crystal\ngraph", LIGHT_BLUE, BLUE, 12)
    box(ax, 0.28, 0.40, 0.22, 0.24, "Shared CGCNN\nbackbone\n3 graph convolutions", LIGHT_BLUE, BLUE, 11)
    box(ax, 0.56, 0.43, 0.16, 0.18, "Shared\nrepresentation", LIGHT_ORANGE, ORANGE, 11)
    box(ax, 0.78, 0.43, 0.16, 0.18, "Linear layer\n128 -> K", LIGHT_GREEN, GREEN, 11)
    arrow(ax, (0.225, 0.52), (0.275, 0.52))
    arrow(ax, (0.505, 0.52), (0.555, 0.52))
    arrow(ax, (0.725, 0.52), (0.775, 0.52))
    task_y = [0.77, 0.53, 0.29]
    labels = ["Formation energy", "Band gap", "Fermi energy"]
    for y, label in zip(task_y, labels):
        box(ax, 0.72, y - 0.055, 0.24, 0.11, label, LIGHT_GREEN, GREEN, 10)
        arrow(ax, (0.86, 0.61), (0.84, y), color=GREEN, mutation_scale=12)
    ax.text(0.5, 0.94, "Shared representation, multiple regression outputs", ha="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.10, "The graph convolution backbone is shared; the last linear layer has one output row per task.", ha="center", fontsize=10, color=GRAY)
    save(fig, "multitask_model.png")


def deployment_flow():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    clean(ax)
    steps = [
        ("Source code", BLUE, LIGHT_BLUE),
        ("Conda\nPython 3.10", BLUE, LIGHT_BLUE),
        ("PyTorch CUDA\n2.7.0+cu128", ORANGE, LIGHT_ORANGE),
        ("Dataset\nCIF + CSV + JSON", GREEN, LIGHT_GREEN),
        ("Train\nmain.py", BLUE, LIGHT_BLUE),
        ("Checkpoint\n+ test CSV", ORANGE, LIGHT_ORANGE),
        ("Predict\npredict.py", GREEN, LIGHT_GREEN),
    ]
    coords = [(0.05, 0.60), (0.29, 0.60), (0.53, 0.60), (0.77, 0.60),
              (0.77, 0.24), (0.41, 0.24), (0.05, 0.24)]
    for (x, y), (label, edge, face) in zip(coords, steps):
        box(ax, x, y, 0.18, 0.18, label, face, edge, 10)
    for i in range(3):
        x, y = coords[i]
        nx, ny = coords[i + 1]
        arrow(ax, (x + 0.18, y + 0.09), (nx, ny + 0.09))
    arrow(ax, (0.86, 0.60), (0.86, 0.42))
    arrow(ax, (0.77, 0.33), (0.59, 0.33))
    arrow(ax, (0.41, 0.33), (0.23, 0.33))
    ax.text(0.5, 0.91, "CGCNN deployment and reproduction workflow", ha="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.08, "Environment is verified before training; outputs are saved in isolated run directories.", ha="center", fontsize=10, color=GRAY)
    save(fig, "deployment_flow.png")


def single_vs_multitask():
    fig, ax = plt.subplots(figsize=(10, 5))
    clean(ax)
    box(ax, 0.04, 0.54, 0.41, 0.28, "Original single-task\n\nid,property\nLinear(128, 1)\nOutput [N, 1]", LIGHT_BLUE, BLUE, 12)
    box(ax, 0.55, 0.54, 0.41, 0.28, "Multi-task extension\n\nid,p1,p2,p3\nLinear(128, 3)\nOutput [N, 3]", LIGHT_ORANGE, ORANGE, 12)
    box(ax, 0.04, 0.18, 0.92, 0.18, "Unchanged: crystal graph construction, ConvLayer, pooling, shared representation", LIGHT_GREEN, GREEN, 12)
    arrow(ax, (0.46, 0.68), (0.54, 0.68), color=GRAY, mutation_scale=18)
    ax.text(0.5, 0.91, "Code-level change: output dimension and target handling", ha="center", fontsize=15, weight="bold")
    save(fig, "single_vs_multitask.png")


def real_metrics():
    metrics_path = ROOT / "results" / "multitask" / "metrics.json"
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    tasks = list(data["tasks"].items())
    names = [name.replace("_", "\n") for name, _ in tasks]
    mae = [item["mae"] for _, item in tasks]
    r2 = [item["r2"] for _, item in tasks]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ["#2f6fbb", "#e58b32", "#3c8b68"]
    axes[0].bar(names, mae, color=colors)
    axes[0].set_title("Test MAE by task")
    axes[0].set_ylabel("MAE")
    axes[0].grid(axis="y", alpha=0.2)
    for i, value in enumerate(mae):
        axes[0].text(i, value + max(mae) * 0.02, f"{value:.4f}", ha="center", fontsize=10)

    axes[1].bar(names, r2, color=colors)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Test R$^2$ by task")
    axes[1].set_ylabel("R$^2$")
    axes[1].grid(axis="y", alpha=0.2)
    for i, value in enumerate(r2):
        axes[1].text(i, value + 0.02, f"{value:.4f}", ha="center", fontsize=10)

    sha = data["source_sha256"][:12]
    fig.suptitle("Real multitask CGCNN test metrics", fontsize=15, weight="bold")
    fig.text(0.5, 0.01, f"Source: {data['rows']} test rows; CSV SHA-256 prefix {sha}", ha="center", fontsize=9, color=GRAY)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    save(fig, "real_multitask_metrics.png")


def main():
    cgcnn_pipeline()
    crystal_graph()
    multitask_model()
    deployment_flow()
    single_vs_multitask()
    real_metrics()


if __name__ == "__main__":
    main()
