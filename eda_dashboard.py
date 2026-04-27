# ============================================================
#  EDA Dashboard — Video Game Sales Dataset
#  Skills: pandas, matplotlib, seaborn
#  Run: python eda_dashboard.py
# ============================================================

# ── 1. IMPORTS ───────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── 2. LOAD DATA ─────────────────────────────────────────────
df = pd.read_csv("video_games.csv")

# ── 3. FIRST LOOK ────────────────────────────────────────────
print("=" * 50)
print("SHAPE  →", df.shape)           # (rows, columns)
print("\nCOLUMNS:\n", df.columns.tolist())
print("\nFIRST 5 ROWS:")
print(df.head())
print("\nDATA TYPES:\n", df.dtypes)
print("\nMISSING VALUES:\n", df.isnull().sum())

# ── 4. SUMMARY STATS ─────────────────────────────────────────
print("\n" + "=" * 50)
print("SUMMARY STATISTICS:")
print(df.describe())                  # mean, std, min, max, etc.

# Extra aggregations
print("\nAvg Critic Score by Genre:")
print(df.groupby("genre")["critic_score"].mean().sort_values(ascending=False))

print("\nTotal Sales by Platform (millions):")
print(df.groupby("platform")["global_sales_millions"].sum().sort_values(ascending=False))

# ── 5. VISUALISATIONS ────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Video Game EDA Dashboard", fontsize=20, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# --- Chart 1: Games per Genre (bar) --------------------------
ax1 = fig.add_subplot(gs[0, 0])
genre_counts = df["genre"].value_counts()
sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=ax1, palette="Blues_r")
ax1.set_title("Games per Genre")
ax1.set_xlabel("Count")
ax1.set_ylabel("")

# --- Chart 2: Avg Critic Score by Genre (horizontal bar) -----
ax2 = fig.add_subplot(gs[0, 1])
avg_score = df.groupby("genre")["critic_score"].mean().sort_values()
sns.barplot(x=avg_score.values, y=avg_score.index, ax=ax2, palette="Greens_r")
ax2.set_title("Avg Critic Score by Genre")
ax2.set_xlabel("Score (0–100)")
ax2.set_ylabel("")

# --- Chart 3: Sales by Platform (bar) ------------------------
ax3 = fig.add_subplot(gs[0, 2])
plat_sales = df.groupby("platform")["global_sales_millions"].sum().sort_values(ascending=False)
sns.barplot(x=plat_sales.index, y=plat_sales.values, ax=ax3, palette="Oranges_r")
ax3.set_title("Total Sales by Platform")
ax3.set_ylabel("Sales (millions)")
ax3.set_xlabel("")
ax3.tick_params(axis="x", rotation=15)

# --- Chart 4: Distribution of Critic Scores (histogram) ------
ax4 = fig.add_subplot(gs[1, 0])
sns.histplot(df["critic_score"], bins=20, kde=True, ax=ax4, color="steelblue")
ax4.set_title("Critic Score Distribution")
ax4.set_xlabel("Score")

# --- Chart 5: Critic Score vs Sales (scatter) ----------------
ax5 = fig.add_subplot(gs[1, 1])
sns.scatterplot(data=df, x="critic_score", y="global_sales_millions",
                hue="genre", alpha=0.6, ax=ax5, legend=False)
ax5.set_title("Critic Score vs Global Sales")
ax5.set_xlabel("Critic Score")
ax5.set_ylabel("Sales (M)")

# --- Chart 6: Critic Score vs User Score (scatter) -----------
ax6 = fig.add_subplot(gs[1, 2])
sns.scatterplot(data=df, x="critic_score", y="user_score",
                hue="platform", alpha=0.7, ax=ax6)
ax6.set_title("Critic Score vs User Score")
ax6.set_xlabel("Critic Score")
ax6.set_ylabel("User Score (0–10)")
ax6.legend(fontsize=7, loc="upper left")

# --- Chart 7: Games released per Year (line) -----------------
ax7 = fig.add_subplot(gs[2, 0])
yearly = df.groupby("year").size()
ax7.plot(yearly.index, yearly.values, marker="o", color="coral")
ax7.set_title("Games Released per Year")
ax7.set_xlabel("Year")
ax7.set_ylabel("Count")

# --- Chart 8: Multiplayer vs Avg Sales (bar) -----------------
ax8 = fig.add_subplot(gs[2, 1])
multi_sales = df.groupby("multiplayer")["global_sales_millions"].mean()
sns.barplot(x=multi_sales.index, y=multi_sales.values, ax=ax8,
            palette=["#e07b54", "#5b8db8"])
ax8.set_title("Multiplayer vs Avg Sales")
ax8.set_ylabel("Avg Sales (M)")
ax8.set_xlabel("Has Multiplayer?")

# --- Chart 9: Correlation Heatmap ----------------------------
ax9 = fig.add_subplot(gs[2, 2])
num_cols = ["critic_score", "user_score", "global_sales_millions", "na_sales", "eu_sales", "year"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax9, linewidths=0.5, annot_kws={"size": 7})
ax9.set_title("Correlation Heatmap")
ax9.tick_params(axis="x", rotation=30, labelsize=7)
ax9.tick_params(axis="y", rotation=0, labelsize=7)

# ── 6. SAVE & SHOW ───────────────────────────────────────────
plt.savefig("eda_dashboard.png", dpi=150, bbox_inches="tight")
print("\n✅ Chart saved as eda_dashboard.png")
plt.show()
