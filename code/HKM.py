import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
class HierarchicalKMeans:
    def __init__(self, branching_factor=10, depth=3):
        self.branching_factor = branching_factor
        self.depth = depth
        self.tree = []

    def fit(self, data):
        def recursive_kmeans(data, depth):
            if depth == 0 or len(data) == 0:
                return []
            kmeans = KMeans(n_clusters=self.branching_factor, random_state=0).fit(data)
            self.tree.append(kmeans.cluster_centers_)
            clusters = []
            for i in range(self.branching_factor):
                cluster_data = data[kmeans.labels_ == i]
                clusters.extend(recursive_kmeans(cluster_data, depth - 1))
            return clusters

        return recursive_kmeans(data, self.depth)

# 加载特征
features = np.load('image_features_list.npy').astype('float32')

# 使用分层K均值聚类来构建视觉词汇表
branching_factor = 10  # 每层的分支因子
depth = 3  # 树的层级深度
hkm = HierarchicalKMeans(branching_factor=branching_factor, depth=depth)
visual_words = hkm.fit(features)

# 将所有层的聚类中心合并为一个数组
visual_words = np.vstack(hkm.tree)
vocabulary_size=500
# 将视觉词汇表截断为所需大小
visual_words = visual_words[:vocabulary_size]
# 保存视觉词汇表
np.save('visual_words_hkm_kmeans.npy', visual_words)

print("视觉词汇树构建完成，词汇大小为:", len(visual_words))