"""Графики для словаря результатов run_benchmark()."""

import matplotlib.pyplot as plt


def plot_benchmark(results, baseline="pytorch", title="Сравнение реализаций"):
    names = list(results)
    labels = {"pytorch": "PyTorch", "triton": "Triton", "cuda": "CUDA C++"}
    palette = {"pytorch": "#64748b", "triton": "#0d9488", "cuda": "#6366f1"}
    colors = [palette.get(name, "#0284c7") for name in names]
    medians = [results[name]["median_us"] for name in names]
    p20 = [results[name]["p20_us"] for name in names]
    p80 = [results[name]["p80_us"] for name in names]
    errors = [
        [median - low for median, low in zip(medians, p20)],
        [high - median for median, high in zip(medians, p80)],
    ]
    speedups = [results[baseline]["median_us"] / value for value in medians]
    rows = list(range(len(names)))
    baseline_label = labels.get(baseline, baseline)

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        fig, (ax_time, ax_speed) = plt.subplots(
            1, 2, figsize=(12, max(4.3, 0.65 * len(names) + 2.3)),
            sharey=True, facecolor="white",
        )
        for ax in (ax_time, ax_speed):
            ax.set_facecolor("white")
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#cbd5e1")
            ax.tick_params(axis="both", length=0, colors="#475569", pad=8)
            ax.margins(y=0.2)

        ax_time.barh(
            rows, medians, height=0.52, color=colors, xerr=errors,
            error_kw={"ecolor": "#0f172a", "capsize": 4, "elinewidth": 1.4},
        )
        ax_time.set_yticks(rows, [labels.get(name, name) for name in names])
        ax_time.invert_yaxis()
        ax_time.set_xlim(0, max(p80) * 1.28)
        ax_time.set_title("Время выполнения", loc="left", pad=16)
        ax_time.set_xlabel("Медиана, мкс", labelpad=10)
        for row, median, high in zip(rows, medians, p80):
            ax_time.annotate(
                f"{median:.2f}", (high, row), xytext=(8, 0),
                textcoords="offset points", va="center", color="#0f172a",
            )

        ax_speed.barh(rows, speedups, height=0.52, color=colors)
        ax_speed.axvline(1, color="#0f172a", linewidth=1.2, linestyle="--")
        ax_speed.set_xlim(0, max(1, max(speedups)) * 1.28)
        ax_speed.set_title("Ускорение", loc="left", pad=16)
        ax_speed.set_xlabel(f"Медиана {baseline_label} / медиана реализации", labelpad=10)
        for row, speedup in zip(rows, speedups):
            ax_speed.annotate(
                f"{speedup:.2f}×", (speedup, row), xytext=(8, 0),
                textcoords="offset points", va="center", color="#0f172a",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
            )

        fig.suptitle(title, x=0.12, y=0.98, ha="left", fontsize=15, fontweight="bold")
        fig.text(
            0.12, 0.025, "",
            fontsize=10, color="#64748b",
        )
        fig.subplots_adjust(left=0.12, right=0.98, top=0.79, bottom=0.23, wspace=0.22)
    return fig
