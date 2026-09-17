#!/usr/bin/env python
"""Build real-data figures for the MP CGCNN presentation."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'results' / 'mp_presentation_20260917'
OUT = ROOT / 'figures' / 'mp_report'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.labelcolor'] = '#334155'
plt.rcParams['xtick.color'] = '#475569'
plt.rcParams['ytick.color'] = '#475569'
plt.rcParams['text.color'] = '#0F172A'
BLUE = '#2F6FBB'
ORANGE = '#D97706'
GREEN = '#2F855A'
RED = '#B91C1C'


def read_epoch_metrics(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def plot_training_pair(csv_path, output_path, title):
    rows = read_epoch_metrics(csv_path)
    epochs = [int(row['epoch']) for row in rows]
    train_loss = [float(row['train_loss']) for row in rows]
    val_loss = [float(row['val_loss']) for row in rows]
    train_mae = [float(row['train_mae']) for row in rows]
    val_mae = [float(row['val_mae']) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))
    axes[0].plot(epochs, train_loss, color=BLUE, lw=2.5, label='Train Loss')
    axes[0].plot(epochs, val_loss, color=ORANGE, lw=2.5, label='Validation Loss')
    axes[0].set_title('Loss', fontsize=15, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].grid(alpha=0.22)

    axes[1].plot(epochs, train_mae, color=BLUE, lw=2.5, label='Train MAE')
    axes[1].plot(epochs, val_mae, color=ORANGE, lw=2.5, label='Validation MAE')
    best_idx = int(np.argmin(val_mae))
    axes[1].scatter([epochs[best_idx]], [val_mae[best_idx]], color=RED, s=65, zorder=4)
    axes[1].annotate(
        f'Best val MAE {val_mae[best_idx]:.4f}\nEpoch {epochs[best_idx]}',
        xy=(epochs[best_idx], val_mae[best_idx]),
        xytext=(epochs[best_idx] + 2, val_mae[best_idx] + 0.02),
        fontsize=10,
        color=RED,
        arrowprops={'arrowstyle': '->', 'color': RED, 'lw': 1.3},
    )
    axes[1].set_title('MAE', fontsize=15, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].grid(alpha=0.22)
    for ax in axes:
        ax.legend(frameon=False, loc='upper right')
    fig.suptitle(title, fontsize=17, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def load_predictions(csv_path):
    rows = []
    with Path(csv_path).open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.reader(handle):
            if row:
                rows.append(row)
    ncols = len(rows[0])
    if ncols == 3:
        num_tasks = 1
    elif ncols >= 5 and ncols % 2 == 1:
        num_tasks = (ncols - 1) // 2
    else:
        raise ValueError(f'Unexpected result CSV shape: {ncols}')
    actual = np.array([[float(row[1 + i]) for row in rows] for i in range(num_tasks)])
    predicted = np.array([[float(row[1 + num_tasks + i]) for row in rows] for i in range(num_tasks)])
    ids = [row[0] for row in rows]
    return ids, actual, predicted


def parity_axis(ax, actual, predicted, title, color):
    lower = min(float(np.nanmin(actual)), float(np.nanmin(predicted)))
    upper = max(float(np.nanmax(actual)), float(np.nanmax(predicted)))
    pad = 0.05 * (upper - lower if upper > lower else 1.0)
    lo, hi = lower - pad, upper + pad
    ax.scatter(actual, predicted, s=9, alpha=0.42, color=color, edgecolors='none')
    ax.plot([lo, hi], [lo, hi], color='#64748B', lw=1.5, ls='--')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(alpha=0.18)


def plot_single_parity(csv_path, metrics_path, output_path):
    ids, actual, predicted = load_predictions(csv_path)
    data = json.loads(Path(metrics_path).read_text(encoding='utf-8'))
    metric = data['tasks']['formation_energy_per_atom']
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), gridspec_kw={'width_ratios': [1.15, 1]})
    parity_axis(axes[0], actual[0], predicted[0], 'Formation energy: predicted vs actual', BLUE)
    residual = predicted[0] - actual[0]
    axes[1].hist(residual, bins=70, color=BLUE, alpha=0.82, edgecolor='white', linewidth=0.3)
    axes[1].axvline(0, color='#64748B', lw=1.5, ls='--')
    axes[1].set_title('Residual distribution', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Predicted - actual')
    axes[1].set_ylabel('Count')
    axes[1].grid(axis='y', alpha=0.18)
    fig.suptitle(
        f"MP single-task test set | N={metric['n']:,} | MAE={metric['mae']:.4f} | "
        f"RMSE={metric['rmse']:.4f} | R2={metric['r2']:.4f}",
        fontsize=15,
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def plot_multi_parity(csv_path, metrics_path, output_path):
    ids, actual, predicted = load_predictions(csv_path)
    data = json.loads(Path(metrics_path).read_text(encoding='utf-8'))
    names = ['formation_energy_per_atom', 'band_gap', 'efermi']
    labels = ['Formation energy', 'Band gap', 'Fermi energy']
    colors = [BLUE, ORANGE, GREEN]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for i, (name, label, color) in enumerate(zip(names, labels, colors)):
        parity_axis(axes[i], actual[i], predicted[i], label, color)
        metric = data['tasks'][name]
        axes[i].text(
            0.04,
            0.96,
            f"MAE {metric['mae']:.3f}\nRMSE {metric['rmse']:.3f}\nR2 {metric['r2']:.3f}",
            transform=axes[i].transAxes,
            va='top',
            ha='left',
            fontsize=10,
            color='#0F172A',
            bbox={'boxstyle': 'round,pad=0.35', 'facecolor': 'white', 'edgecolor': '#CBD5E1', 'alpha': 0.92},
        )
    fig.suptitle(
        f"MP multitask test set | N={data['rows']:,} | one shared model, three tasks",
        fontsize=16,
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def plot_multi_metrics(metrics_path, output_path):
    data = json.loads(Path(metrics_path).read_text(encoding='utf-8'))
    names = ['formation_energy_per_atom', 'band_gap', 'efermi']
    labels = ['Formation\nenergy', 'Band gap', 'Fermi\nenergy']
    colors = [BLUE, ORANGE, GREEN]
    mae = [data['tasks'][name]['mae'] for name in names]
    r2 = [data['tasks'][name]['r2'] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    axes[0].bar(labels, mae, color=colors, width=0.62)
    axes[0].set_title('Test MAE by task', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('MAE')
    axes[0].grid(axis='y', alpha=0.18)
    for i, value in enumerate(mae):
        axes[0].text(i, value + max(mae) * 0.025, f'{value:.4f}', ha='center', fontsize=10)
    axes[1].bar(labels, r2, color=colors, width=0.62)
    axes[1].set_title('Test R2 by task', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel('R2')
    axes[1].grid(axis='y', alpha=0.18)
    for i, value in enumerate(r2):
        axes[1].text(i, value + 0.025, f'{value:.4f}', ha='center', fontsize=10)
    fig.suptitle('MP multitask per-task metrics', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def main():
    single = RUNS / 'single_task_30e'
    multi = RUNS / 'multitask_30e'
    plot_training_pair(
        single / 'single_metrics.csv',
        OUT / 'single_training_curves.png',
        'MP single-task CGCNN: real 30-epoch training curves',
    )
    plot_single_parity(
        single / 'test_results.csv',
        single / 'metrics.json',
        OUT / 'single_test_results.png',
    )
    plot_training_pair(
        multi / 'multitask_metrics.csv',
        OUT / 'multitask_training_curves.png',
        'MP multitask CGCNN: real 30-epoch training curves',
    )
    plot_multi_metrics(multi / 'metrics.json', OUT / 'multitask_metric_bars.png')
    plot_multi_parity(
        multi / 'test_results.csv',
        multi / 'metrics.json',
        OUT / 'multitask_test_results.png',
    )
    print('wrote figures to', OUT)


if __name__ == '__main__':
    main()
