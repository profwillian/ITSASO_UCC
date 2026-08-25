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
    / "n_cbh_scalability_vs_rcc.pdf"
)

OUTPUT_PNG = (
    SWEEP_DIR
    / "n_cbh_scalability_vs_rcc.png"
)

CBH_VALUES = [
    10.0,
    20.0,
]


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

    fig, ax = plt.subplots(
        figsize=(3.45, 2.55)
    )

    markers = [
        "o",
        "s",
    ]

    for index, cbh in enumerate(
        CBH_VALUES
    ):

        optimal_values = []
        all_rcc_values = []

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
                ) < 1e-12
            ]

            if not group:
                raise RuntimeError(
                    f"Missing data for "
                    f"N={n}, Cbh={cbh}."
                )

            all_rcc = next(
                row
                for row in group
                if int(
                    row["m_sv"]
                ) == 0
            )

            optimal = min(
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

            t_opt = float(
                optimal[
                    "tmax_model_s"
                ]
            )

            all_rcc_values.append(
                t_rcc
            )

            optimal_values.append(
                t_opt
            )

            print(
                f"N={n:2d} | "
                f"Cbh={cbh:4g} | "
                f"rho*="
                f"{float(optimal['rho_sv']):.4f} | "
                f"T_opt={t_opt:.6f} | "
                f"T_RCC={t_rcc:.6f} | "
                f"delta="
                f"{t_rcc - t_opt:.6f}"
            )

        optimal_line, = ax.plot(
            n_values,
            optimal_values,
            marker=markers[index],
            linewidth=1.5,
            markersize=5,
            label=(
                rf"Optimal, "
                rf"$C_{{bh}}={cbh:g}$"
            ),
        )

        ax.plot(
            n_values,
            all_rcc_values,
            marker=markers[index],
            linewidth=1.2,
            markersize=4,
            linestyle="--",
            color=optimal_line.get_color(),
            label=(
                rf"All-RCC, "
                rf"$C_{{bh}}={cbh:g}$"
            ),
        )

    ax.set_xlabel(
        r"Number of UAVs $N$"
    )

    ax.set_ylabel(
        r"Maximum latency $T_{\max}$ (s)"
    )

    ax.set_xticks(
        n_values
    )

    ax.set_ylim(
        bottom=0
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    ax.legend(
        fontsize=6.2,
        frameon=True,
        loc="upper left",
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
