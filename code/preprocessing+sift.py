import os
import cv2
import numpy as np
import pickle
data_path=r"oxbuild_images"
def get_image_paths(data_dir):
    image_paths=[] #图片地址

    for image_dir in os.listdir(data_dir):
        image_path=os.path.join(data_dir, image_dir) 
        image_paths.append(image_path)
 
    return image_paths
# 加载训练集和测试集
image_paths= get_image_paths(data_path)
print("样本数量：", len(image_paths))
print(image_paths)

sift = cv2.SIFT_create()
# 提取图像特征
def extract_sift_features(image_paths):
    features = []
    for path in image_paths:
        image = cv2.imread(path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        if descriptors is not None:
            features.append(descriptors)
    return features
# 提取图像特征
features = extract_sift_features(image_paths)
# 存储数据
with open('image_features_list.pkl', 'wb') as f:
    pickle.dump(features, f)
# 读取存储的数据
with open('image_features_list.pkl', 'rb') as f:
    loaded_features = pickle.load(f)
# 打印每个元素的大小
for i, feature in enumerate(loaded_features):
    print(f"Loaded feature {i+1} shape: {feature.shape}")


# 提取图像特征(展开)
def extract_sift_features_flatten(image_paths):
    features = []
    for path in image_paths:
        image = cv2.imread(path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        if descriptors is not None:
            features.extend(descriptors)
    return features
# 提取图像特征
features_flatten = extract_sift_features_flatten(image_paths)
np.save('image_features_list.npy', features)