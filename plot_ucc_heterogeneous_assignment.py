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


# =========================================================
# LOAD SIMULATION RESULTS
# =========================================================

with INPUT.open(
    newline="",
    encoding="utf-8",
) as f:
    rows = list(
        csv.DictReader(f)
    )


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

    emulated_values = [
        float(
            row["tmax_emulated_s"]
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

            "tmax_mean_s":
                statistics.mean(
                    emulated_values
                ),

            "tmax_std_s":
                statistics.stdev(
                    emulated_values
                ),

            "tmax_min_s":
                min(
                    emulated_values
                ),

            "tmax_max_s":
                max(
                    emulated_values
                ),
        }
    )


# =========================================================
# SAVE SIMULATION SUMMARY
# =========================================================

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


# =========================================================
# IEEE-LIKE FIGURE STYLE
# =========================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


fig, ax = plt.subplots(
    figsize=(3.5, 2.45)
)


# =========================================================
# DATA
# =========================================================

x = [
    row["sv_uav"]
    for row in summary
]

tmax = [
    row["tmax_mean_s"]
    for row in summary
]

std = [
    row["tmax_std_s"]
    for row in summary
]


# =========================================================
# SIMULATION LINE
# =========================================================

ax.errorbar(
    x,
    tmax,
    yerr=std,
    marker="o",
    markersize=5,
    linewidth=1.5,
    linestyle="-",
    capsize=2.5,
    label="Simulation",
)


# =========================================================
# AXES
# =========================================================

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
    "Workload executed at SV"
)


ax.set_ylabel(
    r"Maximum E2E latency, $T_{\max}$ (s)"
)


ax.set_xlim(
    0.8,
    4.2,
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


# =========================================================
# GRID
# =========================================================

ax.grid(
    axis="y",
    linestyle=":",
    linewidth=0.55,
    alpha=0.55,
)


ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)


# =========================================================
# OUTPUT
# =========================================================

fig.tight_layout(
    pad=0.4
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


# =========================================================
# CONSOLE SUMMARY
# =========================================================

print()
print(
    "SV | Workload | Tmax mean | "
    "Std(ms) | Min | Max"
)

print("-" * 65)

for row in summary:

    print(
        f"{row['sv_uav']:>2} | "
        f"{row['sv_workload_mbit']:.1f} Mbit | "
        f"{row['tmax_mean_s']:.6f} | "
        f"{row['tmax_std_s']*1000:>7.3f} | "
        f"{row['tmax_min_s']:.6f} | "
        f"{row['tmax_max_s']:.6f}"
    )


print()
print(
    f"[OK] {OUTPUT_PDF}"
)

print(
    f"[OK] {OUTPUT_PNG}"
)
