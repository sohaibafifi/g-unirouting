import matplotlib.pyplot as plt
import numpy as np

# Variant names
variants = [
    "vrp",
    "vrpl",
    "ovrp",
    "ovrpl",
    "vrptw",
    "vrpltw",
    "ovrptw",
    "ovrpltw",
    "vrpmb",
    "vrpmbl",
    "vrpb",
    "vrpbl",
    "ovrpmb",
    "ovrpmbl",
    "ovrpb",
    "ovrpbl",
    "vrpmbtw",
    "vrpmbltw",
    "vrpbtw",
    "vrpbltw",
    "ovrpmbtw",
    "ovrpmbltw",
    "ovrpbtw",
    "ovrpbltw",
    "mixed",
]

# TransformerEncoder avg costs (unaugmented vs. augmented)
cost_unaug = [
    11.12,
    11.53,
    7.24,
    7.24,
    17.60,
    17.71,
    7.83,
    7.83,
    9.86,
    10.41,
    11.02,
    11.72,
    6.75,
    6.76,
    7.75,
    7.73,
    17.36,
    17.84,
    19.49,
    19.94,
    7.64,
    7.60,
    8.43,
    8.43,
    11.28,
]
cost_aug = [
    10.61,
    10.90,
    6.81,
    6.79,
    16.61,
    16.62,
    7.25,
    7.25,
    9.35,
    9.75,
    10.27,
    10.75,
    6.34,
    6.34,
    7.24,
    7.24,
    16.37,
    16.73,
    18.47,
    18.81,
    7.04,
    7.00,
    7.76,
    7.78,
    10.59,
]

# 1. Bar plot: TransformerEncoder avg cost with and without augmentation
plt.figure(figsize=(14, 6))
x = np.arange(len(variants))
width = 0.35
plt.bar(x - width / 2, cost_unaug, width, label="No Aug")
plt.bar(x + width / 2, cost_aug, width, label="Augmented")
plt.xticks(x, variants, rotation=45, fontsize=12)
plt.ylabel("Avg Cost (50 greedy)")
plt.title("TransformerEncoder: Avg Cost with vs. without Augmentation")
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("avg_cost_comparison_augmentation.png", dpi=300, bbox_inches="tight")


# 2. Prepare data for cost comparison (augmented). Exclude indices where costs are infinite:
#    indices 11 (vrpbl), 18 (vrpbtw), 19 (vrpbltw).
exclude = {11, 18, 19}
cost_variants = [v for i, v in enumerate(variants) if i not in exclude]

# Baseline (HGS/pyvrp) avg cost (augmented) for each included variant:
cost_baseline = [
    11.02,
    11.44,
    6.62,
    6.61,
    16.72,
    16.79,
    10.57,
    10.59,
    9.35,
    9.75,
    10.57,
    6.20,
    6.18,
    6.89,
    6.89,
    16.25,
    16.96,
    10.59,
    10.49,
    11.69,
    11.65,
    11.35,
]

# TransformerEncoder avg cost (augmented) for each included variant:
cost_transformer = [
    10.61,
    10.90,
    6.81,
    6.79,
    16.61,
    16.62,
    7.25,
    7.25,
    9.35,
    9.75,
    7.25,
    6.34,
    6.34,
    7.24,
    7.24,
    16.37,
    16.73,
    7.04,
    7.00,
    7.76,
    7.78,
    10.59,
]

# AttentionEncoder avg cost (augmented) for each included variant:
cost_attention = [
    10.64,
    10.92,
    6.84,
    6.83,
    16.60,
    16.63,
    7.26,
    7.25,
    9.39,
    9.78,
    7.26,
    6.38,
    6.38,
    7.27,
    7.28,
    16.35,
    16.71,
    7.05,
    7.02,
    7.78,
    7.80,
    10.61,
]

# 2. Bar plot: Cost comparison (augmented)
plt.figure(figsize=(14, 6))
x2 = np.arange(len(cost_variants))
w2 = 0.25

plt.bar(x2 - w2, cost_baseline, w2, label="HGS (pyvrp)", color="#1f77b4")
plt.bar(x2, cost_transformer, w2, label="Ours", color="#ff7f0e")
plt.bar(x2 + w2, cost_attention, w2, label="Attention", color="#2ca02c")

plt.xticks(x2, cost_variants, rotation=45, fontsize=14)
plt.ylabel("Avg Cost")
plt.title("Cost Comparison")
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("cost_comparison.png", dpi=300, bbox_inches="tight")


# 3. CPU times  for all three methods (25 variants)
pyvrp_cpu_aug = [
    3070.11,
    3124.13,
    2284.70,
    2363.68,
    3115.97,
    2973.32,
    2195.86,
    2206.81,
    2899.58,
    3077.03,
    2206.90,
    2654.94,
    2297.89,
    2210.04,
    1760.66,
    1772.62,
    3177.48,
    3205.33,
    3088.81,
    3083.12,
    2372.69,
    2392.60,
    2412.29,
    2380.35,
    2727.65,
]
transformer_cpu_aug = [
    194.82,
    198.14,
    212.61,
    213.59,
    212.55,
    213.01,
    207.56,
    207.56,
    189.41,
    194.09,
    190.88,
    198.12,
    207.92,
    206.87,
    211.50,
    209.03,
    210.86,
    214.32,
    338.35,
    627.73,
    587.76,
    595.73,
    616.90,
    617.06,
    410.91,
]
attention_cpu_aug = [
    190.86,
    194.63,
    201.61,
    201.64,
    207.60,
    209.56,
    202.26,
    202.92,
    186.08,
    190.42,
    188.95,
    194.78,
    195.63,
    195.79,
    201.05,
    200.79,
    207.21,
    225.74,
    417.73,
    641.34,
    580.60,
    581.77,
    576.81,
    609.31,
    354.80,
]

# 3. Bar plot: CPU time comparison
plt.figure(figsize=(14, 6))
x3 = np.arange(len(variants))
w3 = 0.25

plt.bar(x3 - w3, pyvrp_cpu_aug, w3, label="HGS (pyvrp)", color="#1f77b4")
plt.bar(x3, transformer_cpu_aug, w3, label="Ours", color="#ff7f0e")
plt.bar(x3 + w3, attention_cpu_aug, w3, label="Attention", color="#2ca02c")

plt.xticks(x3, variants, rotation=45, fontsize=14)
plt.ylabel("CPU Time (s)")
plt.xlabel("Variants", fontsize=12)
plt.title("Cost Comparison")
plt.legend(fontsize=12)
plt.title("CPU Time Comparison")
plt.legend(fontsize=12)
# change font of x ticks
plt.tight_layout()
plt.savefig("cpu_time_comparison.png", dpi=300, bbox_inches="tight")
