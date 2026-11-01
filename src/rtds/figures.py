"""The three result figures of the report (Figs. 3-5), written by scripts/analyse.py."""

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
VISIT_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
MODEL_COLORS = {"forest": TEXT, "lstm": "#6250d6"}
MODEL_NAMES = {"forest": "RSF", "lstm": "LSTM"}
RHOS = [0.0, 0.5, 0.9]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "pdf.fonttype": 42,
})
WHITE = {"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"}


def hit_rate_chart(summary, path):
    """Hit rate per rho, one panel per scenario: solid = fixed, dashed = tuned, band = bootstrap interval."""
    with plt.rc_context({**WHITE, "font.size": 8}):
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.4), sharey=True)
        for ax, scen, title in zip(axes, ("A", "B"), ("A: driver = last visit", "B: driver = first visit")):
            for model in ("forest", "lstm"):
                color = MODEL_COLORS[model]
                for settings, style in (("fixed", "-"), ("tuned", "--")):
                    s = summary[(summary["model"] == model) & (summary["settings"] == settings)
                                & (summary["scenario"] == scen)].sort_values("rho")
                    ax.fill_between(RHOS, s["hit_lo"], s["hit_hi"], color=color, alpha=0.15, linewidth=0)
                    ax.plot(RHOS, s["hit_mean"], style, color=color, linewidth=1.4, marker="o", ms=3.5,
                            mfc=color if settings == "fixed" else "white", mew=1.1,
                            label=f"{MODEL_NAMES[model]}, {settings}")
            ax.axhline(1 / 3, color=TEXT_2, linestyle=":", linewidth=1)
            ax.text(0.02, 1 / 3 + 0.015, "chance", color=TEXT_2)
            ax.set_xticks(RHOS, [f"{r:g}" for r in RHOS])
            ax.set_xlabel("correlation ρ", labelpad=1)
            ax.set_xlim(-0.05, 0.95)
            ax.set_ylim(0.2, 1.0)
            ax.set_title(title, loc="left", color=TEXT)
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("Hit rate")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
                   handletextpad=0.4, columnspacing=1.0, handlelength=2.2)
        fig.tight_layout(rect=(0, 0, 1, 0.82))
        fig.savefig(path)
        plt.close(fig)


def per_visit_panels(summary, path, driver_visit):
    """Per visit: share of SurvSHAP(t) credit (Run 3), then the two reliance measures (Run 2)."""
    with plt.rc_context({**WHITE, "font.size": 8}):
        fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.0))
        width = 0.26
        panels = [("share_visit_{k}", "Credit share", "Mean credit share"),
                  ("drop_visit_{k}", "Reliance: C-index drop", "Drop in C-index"),
                  ("ibs_visit_{k}", "Reliance: IBS rise", "Rise in IBS")]
        for ax, (pattern, title, ylabel) in zip(axes, panels):
            for j, k in enumerate((1, 2, 3)):
                col = pattern.format(k=k)
                xs = [x + (j - 1) * (width + 0.02) for x in range(len(summary))]
                mean = summary[f"{col}_mean"]
                ax.bar(xs, mean, width=width, color=VISIT_COLORS[k],
                       label=f"visit {k}" + (" (driver)" if k == driver_visit else ""), zorder=2)
                ax.errorbar(xs, mean, yerr=[mean - summary[f"{col}_lo"], summary[f"{col}_hi"] - mean],
                            fmt="none", ecolor=TEXT, elinewidth=1, capsize=2.5, zorder=3)
            ax.set_xticks(range(len(summary)), [f"ρ = {r:g}" for r in summary["rho"]])
            ax.set_ylabel(ylabel)
            ax.set_title(title, loc="left", color=TEXT)
            ax.axhline(0, color=TEXT_2, linewidth=0.8)
            ax.grid(axis="x", visible=False)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(path)
        plt.close(fig)


def example_curves(items, grid, driver_visit, path):
    """SurvSHAP(t) of each visit over time, one patient per panel. The shaded area is the visit's credit."""
    fig, axes = plt.subplots(1, len(items), figsize=(4.6 * len(items), 4.5))
    c_axes = [ax for ax, it in zip(axes, items) if it["case"] == "C"]
    for ax, it in zip(axes, items):
        for k in (1, 2, 3):
            phi = it["phi"][k - 1]
            name = f"visit {k}" + (" (driver)" if k == driver_visit else "")
            ax.plot(grid, phi, color=VISIT_COLORS[k], linewidth=2, label=name, zorder=3)
            ax.fill_between(grid, 0, phi, color=VISIT_COLORS[k], alpha=0.12, linewidth=0)
        ax.axhline(0, color=TEXT_2, linewidth=0.8)
        share = it["credit"][driver_visit - 1] / it["credit"].sum()
        top = int(np.argmax(it["credit"])) + 1
        ax.set_title(f"{it['case']}. {it['label']}\nρ = {it['rho']:g}, patient {it['patient']}, C-index {it['c_index']:.3f}\n"
                     f"driver share {share:.2f}, top visit {top}",
                     loc="left", color=TEXT, fontsize=9.5)
        ax.set_xlabel("Time")
    axes[0].set_ylabel("SurvSHAP(t): change in survival probability")
    if len(c_axes) == 2:
        lo = min(a.get_ylim()[0] for a in c_axes)
        hi = max(a.get_ylim()[1] for a in c_axes)
        for a in c_axes:
            a.set_ylim(lo, hi)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)
