import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# 聚类特征
def clustering(features, num_clusters, convergence_threshold=0.001, max_iter=1000):
    features = np.array(features)
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=10000, max_iter=max_iter)
    prev_centers = None
    for i in range(max_iter):
        kmeans.partial_fit(features)
        centers = kmeans.cluster_centers_
        if prev_centers is not None and np.linalg.norm(centers - prev_centers) < convergence_threshold:
            break
        prev_centers = centers
    visual_words = kmeans.cluster_centers_
    return visual_words, i+1

# 加载特征
features = np.load('image_features_list.npy').astype('float32')

# 假设我们想要的视觉词汇大小为 5000
num_visual_words = 100000

# 聚类特征以创建视觉词汇
visual_words, iterations = clustering(features, num_visual_words)

print("视觉词汇构建完成，词汇大小为:", len(visual_words))
print("迭代次数:", iterations)

# 保存每张图片的SIFT特征和视觉词汇
np.save('visual_words.npy', visual_words)