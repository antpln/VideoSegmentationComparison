import pandas as pd
import matplotlib.pyplot as plt

# Match IEEE LaTeX font styling (Times family)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
import os
outdir = "figures"
os.makedirs(outdir, exist_ok=True)

# -------------------------------------------------------
# 1. LOAD CSV RESULTS
# -------------------------------------------------------
shire = pd.read_csv("sav_benchmark_summary_shire.csv")
jetson = pd.read_csv("sav_benchmark_summary_jetson.csv")

shire["hardware"] = "RTX 3090"
jetson["hardware"] = "Jetson Orin"

df = pd.concat([shire, jetson], ignore_index=True)

# -------------------------------------------------------
# 2. AVERAGE RESULTS PER (MODEL, HARDWARE)
# -------------------------------------------------------
numeric_cols = df.select_dtypes(include="number").columns

df_avg = df.groupby(["model", "hardware"])[numeric_cols].mean().reset_index()

# Pretty model names
model_map = {
    "sam2_tiny_points": "SAM 2 Tiny",
    "sam2_small_points": "SAM 2 Small",
    "sam2_base_points": "SAM 2 Base",
    "sam2_large_points": "SAM 2 Large",
    "edgetam_points": "EdgeTAM",
}
df_avg["model_pretty"] = df_avg["model"].map(model_map)

# Color palette
color_names = plt.rcParams['axes.prop_cycle'].by_key()['color']
model_colors = {m: color_names[i % len(color_names)]
                for i, m in enumerate(df_avg["model_pretty"].unique())}

# Marker style per hardware platform
hardware_markers = {
    "RTX 3090": "o",
    "Jetson Orin": "^",
}

# -------------------------------------------------------
# 3. FIGURE 1 — FPS VS J&F
# -------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(11, 5.2))

for _, row in df_avg.iterrows():
    ax1.scatter(
        row["fps"], row["JandF"],
        c=model_colors[row["model_pretty"]],
        marker=hardware_markers[row["hardware"]],
        s=90,
        zorder=3
    )

ax1.set_xlabel("FPS (avg per run)")
ax1.set_ylabel("J&F Overlap (avg per run)")
ax1.set_title("Throughput vs Accuracy (Averaged)", fontsize=14)
ax1.grid(True, linestyle="--", alpha=0.3)
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)

# Build color legend + hardware legend
model_handles = [
    plt.Line2D([0], [0], marker='o', linestyle='none',
               markerfacecolor=model_colors[m], markersize=10, label=m)
    for m in model_colors
]
hardware_handles = [
    plt.Line2D(
        [0], [0],
        marker=hardware_markers[h],
        linestyle='none',
        markersize=8,
        markerfacecolor="black",
        markeredgecolor="black",
        markeredgewidth=0.0,
        label=h
    )
    for h in hardware_markers
]

legend1 = ax1.legend(
    handles=model_handles,
    title="Models",
    loc="center left",
    bbox_to_anchor=(1.02, 0.65),
    borderaxespad=0,
    frameon=True,
    fancybox=True
)
legend1.get_frame().set_facecolor("white")
legend1.get_frame().set_alpha(0.92)
legend1.get_frame().set_edgecolor("#999999")
ax1.add_artist(legend1)
ax1.legend(
    handles=hardware_handles,
    title="Hardware",
    loc="center left",
    bbox_to_anchor=(1.02, 0.25),
    borderaxespad=0,
    frameon=True,
    fancybox=True
)
ax1.get_legend().get_frame().set_facecolor("white")
ax1.get_legend().get_frame().set_alpha(0.92)
ax1.get_legend().get_frame().set_edgecolor("#999999")

fig1.tight_layout(rect=[0, 0, 0.78, 1])

# -------------------------------------------------------
# 4. FIGURE 2 — GPU vs CPU footprint
# -------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(11, 5.2))

for _, row in df_avg.iterrows():
    ax2.scatter(
        row["gpu_peak_alloc_MiB"], row["cpu_peak_rss_MiB"],
        c=model_colors[row["model_pretty"]],
        marker=hardware_markers[row["hardware"]],
        s=90,
        zorder=3
    )

ax2.set_xlabel("GPU peak alloc (MiB, avg)")
ax2.set_ylabel("CPU peak RSS (MiB, avg)")
ax2.set_title("GPU vs CPU Memory Footprint (Averaged)", fontsize=14)
ax2.grid(True, linestyle="--", alpha=0.3)
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)

legend1 = ax2.legend(
    handles=model_handles,
    title="Models",
    loc="center left",
    bbox_to_anchor=(1.02, 0.65),
    borderaxespad=0,
    frameon=True,
    fancybox=True
)
legend1.get_frame().set_facecolor("white")
legend1.get_frame().set_alpha(0.92)
legend1.get_frame().set_edgecolor("#999999")
ax2.add_artist(legend1)
ax2.legend(
    handles=hardware_handles,
    title="Hardware",
    loc="center left",
    bbox_to_anchor=(1.02, 0.25),
    borderaxespad=0,
    frameon=True,
    fancybox=True
)
ax2.get_legend().get_frame().set_facecolor("white")
ax2.get_legend().get_frame().set_alpha(0.92)
ax2.get_legend().get_frame().set_edgecolor("#999999")

fig2.tight_layout(rect=[0, 0, 0.78, 1])

plt.show()

# -------------------------------------------------------
# SAVE FIGURES
# -------------------------------------------------------


fig1.savefig(f"{outdir}/fps_vs_jandf.png", dpi=300, bbox_inches="tight")
fig2.savefig(f"{outdir}/gpu_vs_cpu.png", dpi=300, bbox_inches="tight")

# Optional vector outputs (recommended for papers)
fig1.savefig(f"{outdir}/fps_vs_jandf.pdf", bbox_inches="tight")
fig2.savefig(f"{outdir}/gpu_vs_cpu.pdf", bbox_inches="tight")

plt.close(fig1)
plt.close(fig2)
