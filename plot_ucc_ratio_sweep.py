import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path("res_ucc")


def resolve_input_dir(value):
    if value != "latest":
        path = Path(value)

        if not path.exists():
            raise FileNotFoundError(
                f"Input directory not found: {path}"
            )

        return path

    candidates = [
        path
        for path in RESULTS_DIR.glob(
            "sweep_ratio_*"
        )
        if (
            path
            / "ratio_sweep_summary.csv"
        ).exists()
    ]

    if not candidates:
        raise RuntimeError(
            "No ratio sweep results found."
        )

    return max(
        candidates,
        key=lambda path:
            path.stat().st_mtime,
    )


def load_rows(input_dir):
    path = (
        input_dir
        / "ratio_sweep_summary.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Summary file not found: {path}"
        )

    with open(
        path,
        newline="",
        encoding="utf-8",
    ) as csvfile:

        rows = list(
            csv.DictReader(csvfile)
        )

    if not rows:
        raise RuntimeError(
            "Ratio sweep summary is empty."
        )

    return rows


def plot_results(
    rows,
    input_dir,
):
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

    line_styles = [
        "-",
        "--",
        "-.",
        ":",
    ]

    fig, ax = plt.subplots(
        figsize=(3.45, 2.55)
    )

    for index, cbh in enumerate(
        cbh_values
    ):
        selected = [
            row
            for row in rows
            if float(
                row["cbh_mbps"]
            ) == cbh
        ]

        selected.sort(
            key=lambda row:
                float(row["rho_sv"])
        )

        x = [
            float(row["rho_sv"])
            for row in selected
        ]

        y = [
            float(
                row[
                    "tmax_emulated_mean_s"
                ]
            )
            for row in selected
        ]

        ax.plot(
            x,
            y,
            marker=markers[
                index
                % len(markers)
            ],
            linestyle=line_styles[
                index
                % len(line_styles)
            ],
            linewidth=1.15,
            markersize=4.0,
            label=(
                rf"$C_{{bh}}="
                rf"{cbh:g}$ Mbit/s"
            ),
        )

    ax.set_xlabel(
        r"SV execution ratio, "
        r"$\rho_{\mathrm{SV}}$"
    )

    ax.set_ylabel(
        r"Maximum end-to-end latency, "
        r"$T_{\max}$ (s)"
    )

    ax.set_xticks(
        [
            0.00,
            0.25,
            0.50,
            0.75,
            1.00,
        ]
    )

    ax.set_xticklabels(
        [
            "0",
            "0.25",
            "0.50",
            "0.75",
            "1",
        ]
    )

    ax.set_xlim(
        -0.03,
        1.03,
    )

    ax.set_ylim(
        bottom=0,
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.5,
        alpha=0.6,
    )

    ax.legend(
        fontsize=6.5,
        frameon=False,
        loc="lower right",
    )

    ax.tick_params(
        axis="both",
        labelsize=7,
    )

    ax.xaxis.label.set_size(7.5)
    ax.yaxis.label.set_size(7.5)

    fig.tight_layout(
        pad=0.4
    )

    pdf_path = (
        input_dir
        / "tmax_vs_sv_ratio.pdf"
    )

    png_path = (
        input_dir
        / "tmax_vs_sv_ratio.png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return (
        pdf_path,
        png_path,
    )


def print_winners(rows):
    cbh_values = sorted(
        {
            float(row["cbh_mbps"])
            for row in rows
        }
    )

    print("")
    print(
        "[PLOT] ===== MINIMUM LATENCY ====="
    )

    for cbh in cbh_values:
        selected = [
            row
            for row in rows
            if float(
                row["cbh_mbps"]
            ) == cbh
        ]

        winner = min(
            selected,
            key=lambda row:
                float(
                    row[
                        "tmax_emulated_mean_s"
                    ]
                ),
        )

        print(
            f"[PLOT] "
            f"Cbh={cbh:>5.1f} Mbit/s | "
            f"rho_sv="
            f"{float(winner['rho_sv']):.2f} | "
            f"Tmax="
            f"{float(winner['tmax_emulated_mean_s']):.6f} s"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate the UCC SV/RCC ratio "
            "sensitivity figure."
        )
    )

    parser.add_argument(
        "--input",
        default="latest",
        help=(
            "Ratio sweep directory or "
            "'latest'."
        ),
    )

    args = parser.parse_args()

    input_dir = resolve_input_dir(
        args.input
    )

    print(
        f"[PLOT] Input directory: "
        f"{input_dir}"
    )

    rows = load_rows(
        input_dir
    )

    pdf_path, png_path = (
        plot_results(
            rows,
            input_dir,
        )
    )

    print_winners(
        rows
    )

    print("")
    print(
        f"[PLOT] PDF saved to "
        f"{pdf_path}"
    )

    print(
        f"[PLOT] PNG saved to "
        f"{png_path}"
    )


if __name__ == "__main__":
    main()
