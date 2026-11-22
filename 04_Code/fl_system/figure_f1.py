"""Figure F1 of Paper 1 (plan task X-11): (a) the device-gateway-cloud system and its adversaries, (b) the five
defence placements. A drawing: no data, no training. Content follows Sections III and IV (Tables T2 and T3).

Example:
    python figure_f1.py --out 07_Paper/figures
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "savefig.bbox": "tight"})
INK, GREY, LIGHT = "#222222", "#888888", "#F2F2F2"
NOISE = "#E69F00"     # where the DP noise is added (Okabe-Ito orange)
SUSPECT = "#D55E00"   # a party that may be malicious (vermillion)
TRUSTED = "#0072B2"   # a party trusted with the updates (blue)
BAD_DEVICE = "#CC79A7"


def box(ax, x, y, w, h, text="", fc="white", ec=INK, lw=0.8, ls="-", fs=7, weight="normal", color=INK, r=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, color=color,
                linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, color=INK, lw=0.8, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def noise_mark(ax, x, y):
    ax.add_patch(Circle((x, y), 0.9, fc=NOISE, ec="none"))
    ax.text(x, y, "+", ha="center", va="center", fontsize=6, color="white", weight="bold")


def panel_a(ax):
    ax.set_xlim(0, 44)
    ax.set_ylim(0, 60)
    ax.axis("off")
    ax.text(0, 58.5, "(a) System and adversaries", fontsize=8, weight="bold", va="top")
    box(ax, 3, 44.5, 38, 9.5, "Cloud (hospital consortium)\nsigns the model, keeps the registry\nhonest-but-curious",
        fc=LIGHT, fs=6.6)
    xs, labels = [2, 17, 32], ["Gateway 1", "Gateway 2", "Gateway 7"]
    for x, lab in zip(xs, labels):
        box(ax, x, 28, 10, 6, lab, ec=SUSPECT, lw=1.1, ls="--", fs=6.8)
        arrow(ax, x + 3, 34.2, x + 3, 44.3)                 # updates up
        arrow(ax, x + 7, 44.3, x + 7, 34.2, color=GREY)    # signed model down
        for j in range(4):
            cx = x + 1.2 + j * 2.55
            bad = (x, j) in {(2, 2), (32, 0)}
            ax.add_patch(Circle((cx, 16), 0.95, fc=BAD_DEVICE if bad else "white", ec=INK, lw=0.7))
            arrow(ax, cx, 17.0, x + 5 + (cx - x - 5) * 0.4, 27.8, lw=0.6)
        ax.text(x + 5, 13.2, "…", ha="center", va="center", fontsize=8)
    ax.text(29.5, 31, "…", ha="center", va="center", fontsize=9)
    ax.text(33.8, 39.4, "updates", rotation=90, ha="center", va="center", fontsize=5.8)
    ax.text(41.3, 39.4, "signed model", rotation=90, ha="center", va="center", fontsize=5.8, color=GREY)
    ax.text(22, 10.2, "30 ward devices per gateway, 210 in all", ha="center", va="center", fontsize=6.4)
    ax.add_patch(Circle((2.2, 6.3), 0.95, fc=BAD_DEVICE, ec=INK, lw=0.7))
    ax.text(4.2, 6.3, "malicious device (poisoned updates)", va="center", fontsize=6.3)
    ax.add_patch(FancyBboxPatch((0.6, 2.6), 3.2, 1.9, boxstyle="round,pad=0,rounding_size=0.3", fc="white",
                                ec=SUSPECT, lw=1.1, ls="--"))
    ax.text(4.2, 3.55, "suspect gateway: curious, or tampers", va="center", fontsize=6.3)
    ax.text(4.2, 1.0, "(scale, drop, inject, sign flip, Sybil, abort)", va="center", fontsize=6.0, color=SUSPECT)


ROWS = [
    # name, device text, noise at device, link 1 (what the gateway sees), gateway text, gateway trusted, noise at gw,
    # link 2 (uplink), cloud text
    ("P-TG", "DP-SGD", True, "single noised\nupdates", "median", True, False, "1 vector", "median"),
    ("P-TG-c", "clip only", False, "clipped\ngradients", "sum + noise once\n(central DP-SGD)", True, True,
     "1 vector", "median"),
    ("P-PE", "DP-SGD", True, "single noised\nupdates", "HFLMLD detection,\nmean", False, False,
     "1 vector /\n5 rounds", "HFLMLD rule"),
    ("P-FR", "DP-SGD, clip C,\nsecure agg.", True, "masked\nmessages", "relay only", False, False,
     "all masked\nmessages", "median of all\ngroup sums"),
    ("P-TA", "DP-SGD, clip C,\nsecure agg.", True, "group sums\n(k devices)", "τ norm check +\nmedian of groups",
     False, False, "1 vector", "size-aware\ncheck + median"),
]


def panel_b(ax):
    ax.set_xlim(0, 77)
    ax.set_ylim(0, 60)
    ax.axis("off")
    ax.text(0, 58.5, "(b) The five placements", fontsize=8, weight="bold", va="top")
    cols = {"dev": (7, 15), "gw": (35, 18), "cloud": (63.5, 13.5)}   # x, width
    for key, title in (("dev", "Device"), ("gw", "Gateway"), ("cloud", "Cloud")):
        x, w = cols[key]
        ax.text(x + w / 2, 53.2, title, ha="center", va="center", fontsize=7.5, weight="bold")
    ax.text(28.5, 53.2, "gateway sees", ha="center", va="center", fontsize=6.2, style="italic", color=GREY)
    ax.text(58.2, 53.2, "uplink", ha="center", va="center", fontsize=6.2, style="italic", color=GREY)
    h, gap, top = 7.0, 1.6, 50.6
    for i, (name, dev, dev_noise, link1, gw, trusted, gw_noise, link2, cloud) in enumerate(ROWS):
        y = top - (i + 1) * h - i * gap
        bold = name == "P-TA"
        if bold:
            ax.add_patch(FancyBboxPatch((-0.5, y - 0.7), 77.5, h + 1.4, boxstyle="round,pad=0,rounding_size=0.8",
                                        fc="#EAF2F8", ec="none", zorder=0))
        ax.text(0, y + h / 2, name, va="center", fontsize=7, weight="bold" if bold else "normal")
        x, w = cols["dev"]
        box(ax, x, y, w, h, dev, fs=6.2)
        if dev_noise:
            noise_mark(ax, x + w - 0.3, y + h - 0.3)
        x, w = cols["gw"]
        box(ax, x, y, w, h, gw, fs=6.2, ec=TRUSTED if trusted else SUSPECT, lw=1.1, ls="-" if trusted else "--")
        if gw_noise:
            noise_mark(ax, x + w - 0.3, y + h - 0.3)
        x, w = cols["cloud"]
        box(ax, x, y, w, h, cloud, fs=6.2, fc=LIGHT)
        ym = y + h / 2
        arrow(ax, 22.3, ym - 0.8, 34.7, ym - 0.8, lw=0.7)
        ax.text(28.5, ym - 0.4, link1, ha="center", va="bottom", fontsize=5.8, linespacing=1.0)
        arrow(ax, 53.3, ym - 0.8, 63.2, ym - 0.8, lw=0.7)
        ax.text(58.25, ym - 0.4, link2, ha="center", va="bottom", fontsize=5.8, linespacing=1.0)
    ly = 5.4
    noise_mark(ax, 1.0, ly)
    ax.text(2.4, ly, "DP noise added here", va="center", fontsize=6.2)
    for x0, ec, ls, text in ((24, TRUSTED, "-", "gateway trusted"), (45, SUSPECT, "--", "gateway suspect")):
        ax.add_patch(FancyBboxPatch((x0, ly - 0.95), 3.2, 1.9, boxstyle="round,pad=0,rounding_size=0.3", fc="white",
                                    ec=ec, lw=1.1, ls=ls))
        ax.text(x0 + 4.0, ly, text, va="center", fontsize=6.2)
    ax.text(0, 1.6, "All placements: cloud-signed model. P-FR, P-TA: certified groups of k devices, t = k.",
            va="center", fontsize=6.2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="07_Paper/figures")
    args = ap.parse_args(argv)
    fig = plt.figure(figsize=(7.16, 3.3))  # IEEE two-column width (figure*)
    ax_a = fig.add_axes([0.0, 0.0, 0.36, 1.0])
    ax_b = fig.add_axes([0.385, 0.0, 0.615, 1.0])
    panel_a(ax_a)
    panel_b(ax_b)
    os.makedirs(args.out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"F1_system_placements.{ext}"), dpi=300)
    plt.close(fig)
    print("F1 written to", args.out)


if __name__ == "__main__":
    main()
