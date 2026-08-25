import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SWEEP_DIR = Path(
    "res_ucc/sweep_n_cbh_20260825_161429"
)

INPUT_FILE = (
    SWEEP_DIR
    / "n_cbh_optima.csv"
)

OUTPUT_PDF = (
    SWEEP_DIR
    / "n_cbh_optimal_ratio_heatmap.pdf"
)

OUTPUT_PNG = (
    SWEEP_DIR
    / "n_cbh_optimal_ratio_heatmap.png"
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

    matrix = np.full(
        (
            len(n_values),
            len(cbh_values),
        ),
        np.nan,
    )

    for row in rows:

        n = int(
            row["num_uavs"]
        )

        cbh = float(
            row["cbh_mbps"]
        )

        rho = float(
            row["model_rho_sv_star"]
        )

        i = n_values.index(n)
        j = cbh_values.index(cbh)

        matrix[i, j] = rho

    fig, ax = plt.subplots(
        figsize=(3.45, 2.65)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=0.5,
        cmap="cividis",
    )

    ax.set_xticks(
        np.arange(
            len(cbh_values)
        )
    )

    ax.set_xticklabels(
        [
            f"{value:g}"
            for value
            in cbh_values
        ]
    )

    ax.set_yticks(
        np.arange(
            len(n_values)
        )
    )

    ax.set_yticklabels(
        [
            str(value)
            for value
            in n_values
        ]
    )

    ax.set_xlabel(
        r"Backhaul capacity $C_{bh}$ (Mbit/s)"
    )

    ax.set_ylabel(
        r"Number of UAVs $N$"
    )

    for i, n in enumerate(
        n_values
    ):

        for j, cbh in enumerate(
            cbh_values
        ):

            value = matrix[i, j]

            label = (
                f"{value:.4f}"
                .rstrip("0")
                .rstrip(".")
            )

            # The containerized measurements for
            # N=16, Cbh=5 showed a near-equivalent
            # transition between rho=0.25 and 0.3125.
            if (
                n == 16
                and abs(cbh - 5.0) < 1e-12
            ):
                label += "*"

            text_color = (
                "white"
                if value < 0.25
                else "black"
            )

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.05,
        pad=0.04,
    )

    colorbar.set_label(
        r"Optimal SV execution ratio "
        r"$\rho_{\mathrm{SV}}^\star$",
        fontsize=8,
    )

    colorbar.ax.tick_params(
        labelsize=7
    )

    ax.tick_params(
        labelsize=8
    )

    for spine in ax.spines.values():
        spine.set_linewidth(
            0.6
        )

    fig.tight_layout(
        pad=0.5
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

    print("")
    print(
        "[PLOT] * N=16, Cbh=5 Mbit/s: "
        "near-equivalent measured region "
        "between rho_sv=0.25 and 0.3125."
    )


if __name__ == "__main__":
    main()
