from sklearn.cluster import KMeans
import numpy as np
import pickle

from sliding_window import load_config, get_patch_sizes

if __name__ == "__main__":
    # aspect_ratios, widths = load_config()
    # patch_sizes = get_patch_sizes(aspect_ratios, widths)

    patch_sizes = []

    for height in range(30, 70, 5):
        for width in range(30, 70, 5):
            patch_sizes.append((height, width))

    print("Number of patch sizes: ", len(patch_sizes))
    # print("Patch sizes: ", patch_sizes)

    
    clusters = 30
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=clusters) # Adjust the number of clusters as needed
    kmeans.fit(patch_sizes)

    # Select a representative from each cluster
    unique_labels = np.unique(kmeans.labels_)
    representative_patches = []
    for label in unique_labels:
        members = np.array(patch_sizes)[kmeans.labels_ == label]
        representative = np.mean(members, axis=0).astype(int)  # Using mean as representative
        representative_patches.append(tuple(representative))

    representative_patches = list(set(representative_patches))  # Removing

    print("Number of representative patches: ", len(representative_patches))
    print("Representative patches: ", representative_patches)

    with open(f"top_{clusters}_representative_patches_fixed.pkl", "wb") as f:
        pickle.dump(representative_patches, f)