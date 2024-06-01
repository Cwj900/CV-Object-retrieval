import numpy as np
import pyflann as pf

# 加载之前保存的SIFT特征
features = np.load('image_features_list.npy')

# 初始化聚类中心，随机选择features中的行作为初始中心
num_clusters = 500
random_indices = np.random.choice(len(features), size=num_clusters, replace=False)
visual_words = features[random_indices]

# 初始化FLANN对象
flann = pf.FLANN()

for _ in range(10):  # 迭代10次
    # 在每次迭代开始时，使用FLANN库构建一组随机化的k-d树
    params = flann.build_index(visual_words, algorithm='kdtree', trees=8, checks=16)

    # 找到每个点的最近的聚类中心
    nearest_indices, _ = flann.nn_index(features, num_neighbors=1, checks=params['checks'])

    # 更新聚类中心
    new_centers = []
    for i in range(num_clusters):
        cluster_points = features[nearest_indices.ravel() == i]
        cluster_center = cluster_points.mean(axis=0)
        new_centers.append(cluster_center)
    new_centers = np.array(new_centers)
    
    # 检查中心是否有变化，如果没有则停止迭代
    if np.allclose(visual_words, new_centers, atol=1e-5):
        break

    visual_words = new_centers

print("视觉词汇构建完成，词汇大小为:", len(visual_words))
# 保存视觉词汇
np.save('visual_vocabulary_akm.npy', visual_words)



