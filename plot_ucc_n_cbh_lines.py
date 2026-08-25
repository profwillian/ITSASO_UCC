import csv
from pathlib import Path

import matplotlib.pyplot as plt


SWEEP_DIR = Path(
    "res_ucc/sweep_n_cbh_20260825_161429"
)

INPUT_FILE = (
    SWEEP_DIR
    / "n_cbh_optima.csv"
)

OUTPUT_PDF = (
    SWEEP_DIR
    / "n_cbh_optimal_ratio_lines.pdf"
)

OUTPUT_PNG = (
    SWEEP_DIR
    / "n_cbh_optimal_ratio_lines.png"
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

    # Use equally spaced categorical positions.
    # The actual evaluated Cbh values are shown as labels.
    x_positions = list(
        range(
            len(cbh_values)
        )
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

    for index, n in enumerate(
        n_values
    ):

        n_rows = {
            float(row["cbh_mbps"]):
                float(
                    row[
                        "model_rho_sv_star"
                    ]
                )
            for row in rows
            if int(
                row["num_uavs"]
            ) == n
        }

        rho_values = [
            n_rows[cbh]
            for cbh in cbh_values
        ]

        ax.plot(
            x_positions,
            rho_values,
            marker=markers[
                index
                % len(markers)
            ],
            linewidth=1.3,
            markersize=5,
            label=rf"$N={n}$",
        )

    ax.set_xticks(
        x_positions
    )

    ax.set_xticklabels(
        [
            f"{value:g}"
            for value
            in cbh_values
        ]
    )

    ax.set_xlabel(
        r"Backhaul capacity $C_{bh}$ (Mbit/s)"
    )

    ax.set_ylabel(
        r"Optimal SV execution ratio "
        r"$\rho_{\mathrm{SV}}^\star$"
    )

    ax.set_ylim(
        -0.025,
        0.525,
    )

    ax.set_yticks(
        [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ]
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    ax.legend(
        fontsize=7,
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

    print(
        f"[PLOT] PDF: {OUTPUT_PDF}"
    )

    print(
        f"[PLOT] PNG: {OUTPUT_PNG}"
    )


if __name__ == "__main__":
    main()
