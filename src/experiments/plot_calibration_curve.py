import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_calibration_curve():
    results_dir = Path("results")
    full_file = results_dir / "ptq_full_one" / "ptq_grid.json"
    baseline_file = results_dir / "baseline_fp32.json"

    # Load baseline FP32
    with open(baseline_file, "r") as f:
        baseline = json.load(f)
    baseline_top1 = baseline["top1"]

    # Load INT8 result
    with open(full_file, "r") as f:
        full_data = json.load(f)

    full_run = full_data["runs"][0]
    full_batches = full_run["calibration_batches"]
    full_top1 = full_run["top1"]

    # Compute accuracy loss
    delta = baseline_top1 - full_top1

    # Plot
    plt.figure(figsize=(8, 5))

    # Baseline FP32 (linha)
    plt.axhline(
        y=baseline_top1,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline FP32 ({baseline_top1:.4f})"
    )

    # INT8 point
    plt.scatter(
        full_batches,
        full_top1,
        color="green",
        s=120,
        zorder=5,
        label=f"INT8 (PTQ) ({full_batches} batches: {full_top1:.4f})"
    )

    # Texto da perda (sem seta, mais limpo)
    plt.text(
        full_batches + 1,
        full_top1 - 0.002,
        f"Δ acc: {delta:.4f}",
        fontsize=10
    )

    # Labels e estilo
    plt.title("Comparação de Acurácia: FP32 vs INT8 após PTQ")
    plt.xlabel("Número de Batches de Calibração")
    plt.ylabel("Top-1 Accuracy")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    # Ajuste do zoom automático
    plt.ylim(
        min(full_top1, baseline_top1) - 0.01,
        max(full_top1, baseline_top1) + 0.01
    )

    # Salvar imagem
    output_path = results_dir / "calibration_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gráfico salvo em: {output_path}")


if __name__ == "__main__":
    plot_calibration_curve()