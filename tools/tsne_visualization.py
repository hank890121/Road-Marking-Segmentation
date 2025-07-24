import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

clear_features = np.load("/path/to/your/clear_feature")
night_features = np.load("/path/to/your/night_feature")
rainy_features = np.load("/path/to/your/rainy_feature")

# Convert lists to numpy arrays
features_array = np.concatenate([clear_features, night_features, rainy_features])
labels_array = np.array([0 for _ in clear_features] + [1 for _ in night_features] + [2 for _ in rainy_features])

# Apply t-SNE
tsne = TSNE(n_components=3)
# tsne = TSNE(n_components=3, random_state=6)
tsne_results = tsne.fit_transform(features_array)

# Visualize the t-SNE results
plt.figure(figsize=(5, 4))
scatter = plt.scatter(
    tsne_results[: len(clear_features), 0],
    tsne_results[: len(clear_features), 1],
    c=["tab:blue" for _ in clear_features],
    s=10,
    # label="Clear",
)
scatter = plt.scatter(
    tsne_results[len(clear_features) + len(rainy_features) :, 0],
    tsne_results[len(clear_features) + len(rainy_features) :, 1],
    c=["tab:green" for _ in night_features],
    s=10,
    # label="Night",
)
scatter = plt.scatter(
    tsne_results[len(clear_features) : len(clear_features) + len(rainy_features), 0],
    tsne_results[len(clear_features) : len(clear_features) + len(rainy_features), 1],
    c=["tab:orange" for _ in rainy_features],
    s=10,
    # label="Rainy",
)

plt.axis("off")
plt.show()
