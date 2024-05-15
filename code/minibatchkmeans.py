import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# 聚类特征
def clustering(features, num_clusters):
    features = np.array(features)
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=10000, max_iter=10)
    kmeans.fit(features)
    visual_words = kmeans.cluster_centers_
    return visual_words

# 加载特征
features = np.load(r'D:\CV-Object-retrieval\code\image_features_list.npy').astype('float32')

# 假设我们想要的视觉词汇大小为 5000
num_visual_words = 50000

# 聚类特征以创建视觉词汇
visual_words = clustering(features, num_visual_words)

print("视觉词汇构建完成，词汇大小为:", len(visual_words))
# 保存每张图片的SIFT特征和视觉词汇
np.save('visual_words.npy', visual_words)
