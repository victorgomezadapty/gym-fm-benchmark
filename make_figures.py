"""Draw the comparison charts in figures/ from the JSON files in results/."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "results"); F = os.path.join(H, "figures")
os.makedirs(F, exist_ok=True)
j = lambda p: json.load(open(os.path.join(R, p)))
e1 = j("experiment1/kumo_naive_ctx1000_median.json"); e1b = j("experiment1/kumo_naive_ctx3000_mean.json")
tab = j("experiment2/tabular_and_lightgbm_mean.json"); rel = j("experiment2/relational_mean.json"); med = j("experiment2/regression_median.json")
KUMO, LGB, NAIVE = "#76b900", "#1f77b4", "#bbbbbb"

# 1. churn ROC AUC, all variants
rows = [("Kumo naive\n1 month, 1k rows", e1["kumo_churn_30d"]["roc_auc"], NAIVE),
        ("Kumo naive\n1 month, 3k rows", e1b["kumo_churn_30d"]["roc_auc"], NAIVE),
        ("Kumo Relational\n39 months", rel["churn_30d"]["R_multi"]["roc_auc"], KUMO),
        ("Kumo Relational\n39 months + features", rel["churn_30d"]["R_feat"]["roc_auc"], KUMO),
        ("Kumo Tabular\nsame 2,981 rows", tab["churn_30d"]["T_same"]["roc_auc"], KUMO),
        ("Kumo Tabular\n10k rows", tab["churn_30d"]["T_10k"]["roc_auc"], KUMO),
        ("LightGBM\nsame 2,981 rows", tab["churn_30d"]["LGB_same"]["roc_auc"], LGB),
        ("LightGBM\nall 24,044 rows", tab["churn_30d"]["LGB_full"]["roc_auc"], LGB)]
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.bar(range(len(rows)), [r[1] for r in rows], color=[r[2] for r in rows])
ax.axhline(0.5, ls=":", c="k", lw=1); ax.text(7.45, 0.505, "random", ha="right", fontsize=8)
ref = tab["churn_30d"]["LGB_full"]["roc_auc"]; ax.axhspan(ref - 0.04, ref + 0.04, color=LGB, alpha=0.08)
for i, r in enumerate(rows): ax.text(i, r[1] + 0.008, f"{r[1]:.3f}", ha="center", fontsize=9)
ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
ax.set_ylim(0.45, 0.8); ax.set_ylabel("ROC AUC (higher is better)")
ax.set_title("30 day churn on 300 test members: grey = Experiment 1 (naive), green = Kumo fair, blue = LightGBM\nshaded band = LightGBM full \u00b1 0.04 (noise level)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{F}/fig1_churn_auc.png", dpi=150); plt.close(fig)

# 2. regression MAE (median statistic), payments and visits
names = [("Kumo\nRelational\n39 months", "R_multi"), ("Kumo\nRelational\n+ features", "R_feat"),
         ("Kumo\nTabular\nsame rows", "T_same"), ("Kumo\nTabular\n10k rows", "T_10k"),
         ("LightGBM\n\nsame rows", "LGB_same"), ("LightGBM\n\nall rows", "LGB_full")]
cols = [KUMO, KUMO, KUMO, KUMO, LGB, LGB]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, task, unit, naive, nlab in ((axs[0], "revenue_30d", "SAR", 533.56, "predict 0"), (axs[1], "visits_30d", "visits", 6.01, "predict mean")):
    v = [med[task][k]["mae"] for _, k in names]
    ax.bar(range(len(v)), v, color=cols)
    for i, x in enumerate(v): ax.text(i, x * 1.01, f"{x:.0f}" if unit == "SAR" else f"{x:.2f}", ha="center", fontsize=9)
    ax.axhline(naive, ls=":", c="k", lw=1); ax.text(5.4, naive * 1.01, nlab, ha="right", fontsize=8)
    ax.set_xticks(range(len(v))); ax.set_xticklabels([n for n, _ in names], fontsize=8)
    ax.set_ylabel(f"MAE, {unit} (lower is better)"); ax.set_ylim(0, max(v + [naive]) * 1.15)
    ax.set_title({"revenue_30d": "Payments in the next 30 days", "visits_30d": "Visits in the next 30 days"}[task], fontsize=10)
fig.suptitle("Experiment 2 (fair): same anchors, rows and features; Kumo uses the median statistic, like LightGBM's L1 objective", fontsize=10)
fig.tight_layout(); fig.savefig(f"{F}/fig2_regression_mae.png", dpi=150); plt.close(fig)

# 3. information ladder for churn
steps = ["Naive\n(1 month)", "Same\n39 months", "+ same\nfeatures", "Feature table\n(Tabular, same rows)"]
vals = [e1["kumo_churn_30d"]["roc_auc"], rel["churn_30d"]["R_multi"]["roc_auc"], rel["churn_30d"]["R_feat"]["roc_auc"], tab["churn_30d"]["T_same"]["roc_auc"]]
pr = [e1["kumo_churn_30d"]["pr_auc"], rel["churn_30d"]["R_multi"]["pr_auc"], rel["churn_30d"]["R_feat"]["pr_auc"], tab["churn_30d"]["T_same"]["pr_auc"]]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(steps, vals, "o-", c=KUMO, lw=2, label="Kumo ROC AUC"); ax.plot(steps, pr, "s--", c=KUMO, alpha=0.6, label="Kumo PR AUC")
ax.axhline(tab["churn_30d"]["LGB_same"]["roc_auc"], c=LGB, lw=1.5, label="LightGBM ROC AUC (same rows)")
ax.axhline(tab["churn_30d"]["LGB_same"]["pr_auc"], c=LGB, ls="--", lw=1.5, alpha=0.6, label="LightGBM PR AUC (same rows)")
for x, y in zip(steps, vals): ax.text(x, y + 0.012, f"{y:.2f}", ha="center", fontsize=9)
ax.set_ylim(0.15, 0.8); ax.set_ylabel("score"); ax.legend(fontsize=8, loc="lower right")
ax.set_title("Churn: Kumo closes the gap as it gets the same information as LightGBM", fontsize=10)
fig.tight_layout(); fig.savefig(f"{F}/fig3_information_ladder.png", dpi=150); plt.close(fig)
print("figures written")
