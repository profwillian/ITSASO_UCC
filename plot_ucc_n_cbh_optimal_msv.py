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
    / "n_cbh_optimal_msv.pdf"
)

OUTPUT_PNG = (
    SWEEP_DIR
    / "n_cbh_optimal_msv.png"
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

        cbh_rows = {
            int(row["num_uavs"]):
                int(
                    row[
                        "model_m_sv_star"
                    ]
                )
            for row in rows
            if abs(
                float(
                    row["cbh_mbps"]
                )
                - cbh
            )
            < 1e-12
        }

        m_values = [
            cbh_rows[n]
            for n in n_values
        ]

        ax.plot(
            n_values,
            m_values,
            marker=markers[
                index
                % len(markers)
            ],
            linewidth=1.3,
            markersize=5,
            label=rf"$C_{{bh}}={cbh:g}$ Mbit/s",
        )

    ax.set_xlabel(
        r"Number of UAVs $N$"
    )

    ax.set_ylabel(
        r"Optimal SV workloads "
        r"$m_{\mathrm{SV}}^\star$"
    )

    ax.set_xticks(
        n_values
    )

    ax.set_ylim(
        -0.3,
        8.5,
    )

    ax.set_yticks(
        range(
            0,
            9,
        )
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

    print(
        f"[PLOT] PDF: {OUTPUT_PDF}"
    )

    print(
        f"[PLOT] PNG: {OUTPUT_PNG}"
    )


if __name__ == "__main__":
    main()
