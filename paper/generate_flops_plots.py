import json
import pathlib
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8,
    "figure.dpi": 180,
})


def load_case(folder):
    res = ROOT / "RQ2-bis" / folder / "results"
    runs = pd.read_csv(res / "runs.csv")
    curves = pd.read_csv(res / "curves.csv")
    timing = pd.read_csv(res / "timing.csv")
    cfg_flops = runs[
        ["variant", "K", "width", "depth", "d_s", "dyn_params", "rec_params", "p_total", "flops_per_eval"]
    ].drop_duplicates()
    timing = timing.merge(
        cfg_flops, on=["variant", "K", "width", "depth", "d_s"], how="left"
    )
    t16 = timing[timing["threads"] == 16][
        ["variant", "K", "width", "depth", "d_s", "s_per_eval", "flops_per_eval"]
    ].drop_duplicates()
    runs = runs.merge(
        t16[["variant", "K", "width", "depth", "d_s", "s_per_eval"]],
        on=["variant", "K", "width", "depth", "d_s"],
        how="left",
    )
    curves = curves.merge(
        t16[["variant", "K", "width", "depth", "d_s", "s_per_eval", "flops_per_eval"]],
        on=["variant", "K", "width", "depth", "d_s"],
        how="left",
    )
    curves["flops_cum"] = curves["n_fev"] * curves["flops_per_eval"]
    curves["time_T_cum"] = curves["n_fev"] * curves["s_per_eval"]
    return runs, curves, timing


def anytime_pareto(df, xcol, ycol="nrmse_test"):
    pts = df[[xcol, ycol]].dropna().sort_values(xcol).to_numpy()
    front = []
    best = np.inf
    for x, y in pts:
        if y < best:
            front.append((x, y))
            best = y
    return np.array(front)


runs_B, curves_B, timing_B = load_case("2D-Stokes")
runs_C, curves_C, timing_C = load_case("ADR-1b")

# Figure 1: Continuous FLOPs and Pass-T wall-time trajectories across checkpoints
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

for row, (case_label, curves, runs) in enumerate([
    ("Case B (2D Navier-Stokes cavity)", curves_B, runs_B),
    ("Case C (1D Aliev-Panfilov fronts)", curves_C, runs_C),
]):
    med_c = (
        curves.groupby(["variant", "K", "width", "depth", "d_s", "iter"], as_index=False)
        .agg(
            flops_cum=("flops_cum", "median"),
            time_T_cum=("time_T_cum", "median"),
            nrmse_test=("nrmse_test", "median"),
            nrmse_valid=("nrmse_valid", "median"),
        )
    )
    for col, (xcol, xlabel) in enumerate([
        ("flops_cum", "Cumulative training operations (FLOPs = $n_{\\mathrm{fev}} \\times \\mathrm{FLOPs}_{\\mathrm{eval}}$)"),
        ("time_T_cum", "Cumulative Pass-T training wall-time $n_{\\mathrm{fev}} \\times s_{\\mathrm{eval}}$ [s] (16 threads)"),
    ]):
        ax = axes[row, col]
        for variant, color, marker, label_prefix in [
            ("L0", "#1f77b4", "o", "L0 (nonlinear MLP)"),
            ("L2", "#d62728", "s", "L2 (linear basis)"),
        ]:
            sub = med_c[med_c["variant"] == variant]
            for _, grp in sub.groupby(["K", "width", "depth", "d_s"]):
                grp = grp.sort_values("iter")
                ax.plot(
                    grp[xcol],
                    grp["nrmse_test"],
                    color=color,
                    alpha=0.25,
                    lw=1.0,
                )
                ax.scatter(
                    grp[xcol].iloc[-1],
                    grp["nrmse_test"].iloc[-1],
                    color=color,
                    marker=marker,
                    s=28,
                    alpha=0.85,
                    zorder=4,
                )
            front = anytime_pareto(sub, xcol, "nrmse_test")
            if len(front) > 0:
                ax.step(
                    front[:, 0],
                    front[:, 1],
                    where="post",
                    color=color,
                    lw=2.2,
                     label=f"{label_prefix} anytime Pareto envelope",
                    zorder=5,
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Test NRMSE (median of 3 seeds)")
        ax.set_title(f"{case_label}: Error vs {'FLOPs' if col == 0 else 'Pass-T Wall-Time'}")
        ax.grid(True, which="both", ls=":", alpha=0.45)
        ax.legend(loc="upper right")

fig.tight_layout()
fig.savefig(OUT / "fig_flops_trajectories.png", bbox_inches="tight")
plt.close(fig)

# Figure 2: Arithmetic intensity, GFLOP/s throughput, and decomposed FLOPs
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))

# Panel (a): Decomposed FLOPs per evaluation (dynamics vs reconstructor)
ax = axes[0]
# Case B: S=80, T=101, Nx=200; Case C: S=100, T=501, Nx=20
for case_name, runs, S, T, Nx, ncomp, marker in [
    ("Case B", runs_B, 80, 101, 200, 2, "o"),
    ("Case C", runs_C, 100, 501, 20, 1, "^"),
]:
    cfg = runs.groupby(["variant", "K", "width", "depth", "d_s"], as_index=False).first()
    flops_dyn = 3.0 * 2.0 * cfg["dyn_params"] * S * (T - 1)
    flops_rec = cfg["flops_per_eval"] - flops_dyn
    for variant, color in [("L0", "#1f77b4"), ("L2", "#d62728")]:
        m = cfg["variant"] == variant
        ax.scatter(
            flops_dyn[m],
            flops_rec[m],
            c=color,
            marker=marker,
            s=42,
            alpha=0.85,
            label=f"{case_name} {variant}",
        )
ax.plot([1e6, 1e10], [1e6, 1e10], "k--", lw=1.0, alpha=0.5, label="1:1 parity")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Dynamics rollout FLOPs / eval ($6 P_{\\mathrm{dyn}} S (T-1)$)")
ax.set_ylabel("Reconstructor FLOPs / eval ($3 \\times \\mathrm{FLOPs}_{\\mathrm{rec}}$)")
ax.set_title("(a) Dynamics vs Reconstructor FLOPs per Eval")
ax.grid(True, which="both", ls=":", alpha=0.45)
ax.legend(loc="lower right", fontsize=7.5)

# Panel (b): Sustained GFLOP/s vs FLOPs per eval at 16 threads
ax = axes[1]
for case_name, timing, marker in [("Case B", timing_B, "o"), ("Case C", timing_C, "^")]:
    t16 = timing[timing["threads"] == 16].copy()
    t16["gflops_s"] = (t16["flops_per_eval"] / t16["s_per_eval"]) / 1e9
    for variant, color in [("L0", "#1f77b4"), ("L2", "#d62728")]:
        sub = t16[t16["variant"] == variant]
        ax.scatter(
            sub["flops_per_eval"],
            sub["gflops_s"],
            c=color,
            marker=marker,
            s=44,
            alpha=0.85,
            label=f"{case_name} {variant} (16 thr)",
        )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Analytic FLOPs per evaluation ($\\mathrm{FLOPs}_{\\mathrm{eval}}$)")
ax.set_ylabel("Sustained throughput [GFLOP/s] at 16 threads")
ax.set_title("(b) Hardware Throughput vs Arithmetic Volume")
ax.grid(True, which="both", ls=":", alpha=0.45)
ax.legend(loc="lower right", fontsize=7.5)

# Panel (c): Multi-core GFLOP/s scaling across thread counts p in {1..112}
ax = axes[2]


def parse_threads(folder):
    nb = json.load(open(ROOT / "RQ2-bis" / folder / "executed_timing.ipynb"))
    txt = "\n".join(
        "".join(o.get("text", []))
        for c in nb["cells"]
        for o in c.get("outputs", [])
    )
    rows = [
        (int(m.group(1)), float(m.group(2)))
        for m in re.finditer(r"threads\s+(\d+):\s+([0-9.]+)\s+s/evaluation", txt)
    ]
    return pd.DataFrame(rows, columns=["threads", "s_per_eval"])


for case_name, folder, runs, w_can, L_can, n_can, color, marker in [
    ("Case B L0 canonical ($T=101, N_x=200$)", "2D-Stokes", runs_B, 7, 2, 1, "#1f77b4", "o"),
    ("Case C L0 canonical ($T=501, N_x=20$)", "ADR-1b", runs_C, 8, 1, 12, "#2ca02c", "s"),
]:
    ts = parse_threads(folder).sort_values("threads")
    f_eval = runs[
        (runs["variant"] == "L0")
        & (runs["width"] == w_can)
        & (runs["depth"] == L_can)
        & (runs["d_s"] == n_can)
    ]["flops_per_eval"].iloc[0]
    gflops = (f_eval / ts["s_per_eval"]) / 1e9
    ax.plot(ts["threads"], gflops, marker=marker, color=color, lw=2.0, label=case_name)
    for p_val, gf_val in zip(ts["threads"], gflops):
        print(f"  {case_name[:6]} p={p_val:3d}: {gf_val:.2f} GFLOP/s")
ax.axvline(16, color="k", ls=":", alpha=0.6, label="Selected $p^*=16$ threads")
ax.axvline(56, color="gray", ls="--", alpha=0.5, label="Single-socket boundary (56c)")
ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 112])
ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "112"])
ax.set_xlabel("CPU Threads $p$ (Intel Xeon Platinum 8480+)")
ax.set_ylabel("Sustained throughput [GFLOP/s]")
ax.set_title("(c) Multi-Core GFLOP/s Scaling on Leonardo")
ax.grid(True, which="both", ls=":", alpha=0.45)
ax.legend(loc="upper left", fontsize=7.5)

fig.tight_layout()
fig.savefig(OUT / "fig_flops_arithmetic_intensity.png", bbox_inches="tight")
plt.close(fig)

# Print summary statistics for the LaTeX section
for name, timing, S, T, Nx in [
    ("Case B", timing_B, 80, 101, 200),
    ("Case C", timing_C, 100, 501, 20),
]:
    t16 = timing[timing["threads"] == 16].copy()
    t16["gflops_s"] = (t16["flops_per_eval"] / t16["s_per_eval"]) / 1e9
    t16["flops_dyn"] = 6.0 * t16["dyn_params"] * S * (T - 1)
    t16["dyn_flop_share_%"] = 100.0 * t16["flops_dyn"] / t16["flops_per_eval"]
    print(f"=== {name} (16 threads) ===")
    for v in ["L0", "L2"]:
        sub = t16[t16["variant"] == v]
        print(
            f"  {v}: FLOPs/eval in [{sub['flops_per_eval'].min():.3e}, {sub['flops_per_eval'].max():.3e}], "
            f"GFLOP/s median={sub['gflops_s'].median():.2f} [{sub['gflops_s'].min():.2f}, {sub['gflops_s'].max():.2f}], "
            f"dyn FLOP share in [{sub['dyn_flop_share_%'].min():.3f}%, {sub['dyn_flop_share_%'].max():.3f}%]"
        )
