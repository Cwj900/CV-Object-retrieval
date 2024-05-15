import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
import random

# 构建随机化的k-d树
def build_kd_trees(features, num_trees):
    kd_trees = []
    num_dimensions = features.shape[1]

    for _ in range(num_trees):
        random_dimensions = random.sample(range(num_dimensions), random.randint(5, 15))
        kd_tree = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        kd_tree.fit(features[:, random_dimensions])
        kd_trees.append((kd_tree, random_dimensions))

    return kd_trees

# AKM聚类特征
def approximate_kmeans(features, num_clusters, num_trees=8):
    features = np.array(features)
    kd_trees = build_kd_trees(features, num_trees)

    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=10000, max_iter=10,metric='manhattan')
    kmeans.cluster_centers_ = np.random.permutation(features)[:num_clusters]

    for _ in range(10):  # 假设迭代10次
        closest_centers = [[] for _ in range(num_clusters)]

        for idx, feature in enumerate(features):
            closest_center = None
            min_distance = float('inf')

            for tree, dimensions in kd_trees:
                _, indices = tree.kneighbors(feature[dimensions].reshape(1, -1))

                for index in indices[0]:
                    distance = np.linalg.norm(feature - kmeans.cluster_centers_[index])

                    if distance < min_distance:
                        closest_center = index
                        min_distance = distance

            closest_centers[closest_center].append(idx)

        for j in range(num_clusters):
            if closest_centers[j]:
                kmeans.cluster_centers_[j] = np.mean(features[closest_centers[j]], axis=0)

    return kmeans.cluster_centers_

# 加载特征
features = np.load('code\image_features_list.npy').astype('float32')

# 假设我们想要的视觉词汇大小为 5000
num_visual_words = 50000

# 聚类特征以创建视觉词汇
visual_words = approximate_kmeans(features, num_visual_words)

print("视觉词汇构建完成，词汇大小为:", len(visual_words))
# 保存每张图片的SIFT特征和视觉词汇
np.save('visual_words_AKM.npy', visual_words)
