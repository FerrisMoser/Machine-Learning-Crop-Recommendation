from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plotting)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Run 3D PCA
pca_3d = PCA(n_components=3)
pca_result_3d = pca_3d.fit_transform(X)
df["pca1_3d"] = pca_result_3d[:, 0]
df["pca2_3d"] = pca_result_3d[:, 1]
df["pca3_3d"] = pca_result_3d[:, 2]

# 3D Scatter plot grouped by crop label
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

# Generate consistent color mapping
unique_labels = sorted(df["label"].unique())
palette = sns.color_palette("tab20", len(unique_labels))
label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}

for label in unique_labels:
    crop_df = df[df["label"] == label]
    ax.scatter(
        crop_df["pca1_3d"],
        crop_df["pca2_3d"],
        crop_df["pca3_3d"],
        label=label,
        color=label_to_color[label],
        edgecolor='k',
        s=50,
        alpha=0.75
    )

ax.set_title("🌾 3D PCA Projection of Crops", fontsize=16)
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', title="Crop")
plt.tight_layout()
plt.savefig("Final stuff/pca_3d_crop_projection.png")
plt.close()
