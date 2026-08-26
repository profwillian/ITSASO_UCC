#!/usr/bin/env python3

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


# =========================================================
# FILES
# =========================================================

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

if not INPUT.exists():
    raise RuntimeError(
        f"Input file not found: {INPUT}"
    )


with INPUT.open(
    newline="",
    encoding="utf-8",
) as f:
    rows = list(
        csv.DictReader(f)
    )


if not rows:
    raise RuntimeError(
        "No simulation results found."
    )


# =========================================================
# GROUP BY WORKLOAD EXECUTED AT SV
# =========================================================

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

    group = grouped[
        sv_uav
    ]

    values = [
        float(
            row[
                "tmax_emulated_s"
            ]
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
                len(values),

            "tmax_mean_s":
                statistics.mean(
                    values
                ),

            "tmax_std_s":
                statistics.stdev(
                    values
                ),

            "tmax_min_s":
                min(values),

            "tmax_max_s":
                max(values),
        }
    )


# =========================================================
# SAVE AGGREGATED SIMULATION RESULTS
# =========================================================

SUMMARY.parent.mkdir(
    parents=True,
    exist_ok=True,
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
    writer.writerows(
        summary
    )


# =========================================================
# PUBLICATION STYLE
# =========================================================

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

        "lines.linewidth":
            1.55,

        "pdf.fonttype":
            42,

        "ps.fonttype":
            42,
    }
)


fig, ax = plt.subplots(
    figsize=(3.5, 2.65)
)


# =========================================================
# SERIES
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
    markersize=5.2,
    linewidth=1.55,
    linestyle="-",
    capsize=2.5,
    capthick=0.9,
    elinewidth=0.9,
)


# =========================================================
# VALUE LABELS
# =========================================================

label_offsets = {
    1: (-5, 6),
    2: (0, 6),
    3: (0, 6),
    4: (5, 6),
}

for x_value, y_value in zip(
    x,
    tmax,
):

    offset_x, offset_y = (
        label_offsets[x_value]
    )

    ax.annotate(
        f"{y_value:.3f}",
        xy=(
            x_value,
            y_value,
        ),
        xytext=(
            offset_x,
            offset_y,
        ),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=6.2,
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
    0.75,
    4.25,
)


ax.set_ylim(
    1.35,
    1.525,
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
# FIXED EXPERIMENT PARAMETERS
# =========================================================

parameter_text = (
    r"$N=4$, "
    r"$C_{bh}=5$ Mbit/s, "
    r"$\rho_{SV}=0.25$"
    "\n"
    r"$D_n=[1.5,\,2.0,\,2.5,\,2.0]$ Mbit"
    "\n"
    r"$10$ runs per assignment"
)


ax.text(
    0.975,
    0.965,
    parameter_text,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=6.7,
    bbox={
        "boxstyle":
            "round,pad=0.30",

        "facecolor":
            "white",

        "edgecolor":
            "0.65",

        "linewidth":
            0.6,

        "alpha":
            0.94,
    },
)


# =========================================================
# GRID AND FRAME
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
    pad=0.45
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


plt.close(
    fig
)


# =========================================================
# CONSOLE SUMMARY
# =========================================================

print()
print(
    "SV | Workload | Mean Tmax | "
    "Std(ms) | Min | Max"
)

print("-" * 69)


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
    f"Best Tmax  = "
    f"{best_value:.6f} s"
)

print(
    f"Worst Tmax = "
    f"{worst_value:.6f} s"
)

print(
    f"Span       = "
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
