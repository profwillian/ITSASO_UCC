import csv
from pathlib import Path

import matplotlib.pyplot as plt


SWEEP_DIR = Path(
    "res_ucc/sweep_n_cbh_20260825_161429"
)

INPUT_FILE = (
    SWEEP_DIR
    / "n_cbh_sweep_summary.csv"
)

OUTPUT_PDF = (
    SWEEP_DIR
    / "n_cbh_gain_vs_rcc.pdf"
)

OUTPUT_PNG = (
    SWEEP_DIR
    / "n_cbh_gain_vs_rcc.png"
)


def main():

    with INPUT_FILE.open(
        newline="",
        encoding="utf-8",
    ) as handle:

        rows = list(
            csv.DictReader(handle)
        )

    n_values = sorted(
        {
            int(row["num_uavs"])
            for row in rows
        }
    )

    cbh_values = sorted(
        {
            float(row["cbh_mbps"])
            for row in rows
        }
    )

    markers = [
        "o",
        "s",
        "^",
        "D",
    ]

    fig, ax = plt.subplots(
        figsize=(3.45, 2.55)
    )

    for index, cbh in enumerate(
        cbh_values
    ):

        gains = []

        for n in n_values:

            group = [
                row
                for row in rows
                if int(
                    row["num_uavs"]
                ) == n
                and abs(
                    float(
                        row["cbh_mbps"]
                    )
                    - cbh
                )
                < 1e-12
            ]

            if not group:
                raise RuntimeError(
                    f"Missing results for "
                    f"N={n}, Cbh={cbh}."
                )

            all_rcc = next(
                row
                for row in group
                if int(
                    row["m_sv"]
                ) == 0
            )

            best = min(
                group,
                key=lambda row:
                    float(
                        row["tmax_model_s"]
                    ),
            )

            t_rcc = float(
                all_rcc[
                    "tmax_model_s"
                ]
            )

            t_best = float(
                best[
                    "tmax_model_s"
                ]
            )

            gain_pct = (
                (
                    t_rcc
                    - t_best
                )
                / t_rcc
                * 100.0
            )

            gains.append(
                gain_pct
            )

            print(
                f"N={n:2d} | "
                f"Cbh={cbh:4g} | "
                f"T_RCC={t_rcc:.6f} | "
                f"T_best={t_best:.6f} | "
                f"gain={gain_pct:.3f}%"
            )

        ax.plot(
            n_values,
            gains,
            marker=markers[
                index
                % len(markers)
            ],
            linewidth=1.3,
            markersize=5,
            label=(
                rf"$C_{{bh}}={cbh:g}$ Mbit/s"
            ),
        )

    ax.axhline(
        0.0,
        linewidth=0.7,
        linestyle="--",
    )

    ax.set_xlabel(
        r"Number of UAVs $N$"
    )

    ax.set_ylabel(
        "Latency reduction vs. all-RCC (%)"
    )

    ax.set_xticks(
        n_values
    )

    ax.set_ylim(
        -1.0,
        48.0,
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    ax.legend(
        fontsize=6.5,
        frameon=True,
        ncol=2,
        loc="upper right",
    )

    ax.tick_params(
        labelsize=8
    )

    for spine in ax.spines.values():
        spine.set_linewidth(
            0.7
        )

    fig.tight_layout(
        pad=0.45
    )

    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    print("")
    print(
        f"[PLOT] PDF: {OUTPUT_PDF}"
    )

    print(
        f"[PLOT] PNG: {OUTPUT_PNG}"
    )


if __name__ == "__main__":
    main()
