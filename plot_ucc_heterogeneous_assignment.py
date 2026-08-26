#!/usr/bin/env python3

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


INPUT = Path(
    "paper_results/experiment2/"
    "heterogeneous_assignment_runs.csv"
)

SUMMARY = Path(
    "paper_results/experiment2/"
    "heterogeneous_assignment_summary.csv"
)

OUTPUT_PDF = Path(
    "figures/"
    "ucc_heterogeneous_assignment.pdf"
)

OUTPUT_PNG = Path(
    "figures/"
    "ucc_heterogeneous_assignment.png"
)


def load_rows():
    with INPUT.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(
            csv.DictReader(f)
        )


rows = load_rows()

grouped = {}

for row in rows:

    sv_uav = int(
        row["sv_uav"]
    )

    grouped.setdefault(
        sv_uav,
        [],
    ).append(row)


summary = []

for sv_uav in sorted(grouped):

    group = grouped[sv_uav]

    model_values = [
        float(
            row["tmax_model_s"]
        )
        for row in group
    ]

    emulated_values = [
        float(
            row["tmax_emulated_s"]
        )
        for row in group
    ]

    error_values = [
        float(
            row["tmax_error_pct"]
        )
        for row in group
    ]

    summary.append(
        {
            "sv_uav":
                sv_uav,
            "sv_workload_mbit":
                float(
                    group[0][
                        "sv_workload_mbit"
                    ]
                ),
            "runs":
                len(group),
            "tmax_model_s":
                statistics.mean(
                    model_values
                ),
            "tmax_emulated_mean_s":
                statistics.mean(
                    emulated_values
                ),
            "tmax_emulated_std_s":
                statistics.stdev(
                    emulated_values
                ),
            "tmax_emulated_min_s":
                min(
                    emulated_values
                ),
            "tmax_emulated_max_s":
                max(
                    emulated_values
                ),
            "error_mean_pct":
                statistics.mean(
                    error_values
                ),
        }
    )


with SUMMARY.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(summary)


plt.rcParams.update(
    {
        "font.family":
            "serif",
        "font.size":
            8,
        "axes.labelsize":
            8,
        "xtick.labelsize":
            7.5,
        "ytick.labelsize":
            7.5,
        "legend.fontsize":
            7.5,
        "pdf.fonttype":
            42,
        "ps.fonttype":
            42,
    }
)


fig, ax = plt.subplots(
    figsize=(3.5, 2.55)
)


x = list(
    range(len(summary))
)

offset = 0.065


model = [
    row["tmax_model_s"]
    for row in summary
]

emulated = [
    row["tmax_emulated_mean_s"]
    for row in summary
]

std = [
    row["tmax_emulated_std_s"]
    for row in summary
]


ax.scatter(
    [
        value - offset
        for value in x
    ],
    model,
    marker="s",
    s=31,
    facecolors="white",
    edgecolors="black",
    linewidths=1.0,
    label="Analytical model",
    zorder=3,
)


ax.errorbar(
    [
        value + offset
        for value in x
    ],
    emulated,
    yerr=std,
    fmt="o",
    markersize=4.8,
    capsize=2.5,
    linewidth=0.9,
    color="0.25",
    ecolor="0.35",
    markerfacecolor="0.25",
    markeredgecolor="0.25",
    label=r"Emulation mean $\pm$ 1 SD",
    zorder=4,
)


labels = [
    (
        f"UAV {row['sv_uav']}\n"
        f"{row['sv_workload_mbit']:.1f} Mbit"
    )
    for row in summary
]


ax.set_xticks(
    x,
    labels,
)


ax.set_xlabel(
    "Workload executed at the SV"
)

ax.set_ylabel(
    r"Maximum E2E latency, $T_{\max}$ (s)"
)


ax.set_ylim(
    1.35,
    1.50,
)


ax.set_yticks(
    [
        1.35,
        1.38,
        1.41,
        1.44,
        1.47,
        1.50,
    ]
)


ax.grid(
    axis="y",
    linestyle=":",
    linewidth=0.55,
    alpha=0.6,
)


ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)


ax.legend(
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        1.02,
    ),
    ncol=2,
    frameon=False,
    columnspacing=1.0,
    handletextpad=0.4,
)


fig.tight_layout(
    pad=0.35
)


OUTPUT_PDF.parent.mkdir(
    parents=True,
    exist_ok=True,
)


fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print()
print(
    "SV | Model | Emulated mean | "
    "Std (ms) | Error (%)"
)

print("-" * 59)

for row in summary:

    print(
        f"{row['sv_uav']:>2} | "
        f"{row['tmax_model_s']:.6f} | "
        f"{row['tmax_emulated_mean_s']:.6f} | "
        f"{row['tmax_emulated_std_s']*1000:>8.3f} | "
        f"{row['error_mean_pct']:+.3f}"
    )


best = min(
    summary,
    key=lambda row:
        row["tmax_emulated_mean_s"],
)

worst = max(
    summary,
    key=lambda row:
        row["tmax_emulated_mean_s"],
)


span_ms = (
    worst["tmax_emulated_mean_s"]
    - best["tmax_emulated_mean_s"]
) * 1000.0


print()
print(
    f"Emulated best-worst span = "
    f"{span_ms:.3f} ms"
)

print()
print(
    f"[OK] {OUTPUT_PDF}"
)

print(
    f"[OK] {OUTPUT_PNG}"
)

print(
    f"[OK] {SUMMARY}"
)
