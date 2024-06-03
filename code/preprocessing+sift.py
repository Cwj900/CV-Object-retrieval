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
    keypoints_locations = []  # 新增，用于存储关键点位置信息
    failed_images = []  # 存储提取失败的图像路径
    for path in image_paths:
        image = cv2.imread(path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        if descriptors is not None:
            features.append(descriptors)
            keypoints_locations.append([kp.pt for kp in keypoints])  
        else:
            failed_images.append(path)  # 将提取失败的图像路径添加到列表中
    return features,keypoints_locations, failed_images

# 提取图像特征和确认提取失败的图像
features,keypoints_locations, failed_images = extract_sift_features(image_paths)

def save_features_and_positions(features,  keypoints_locations, file_name):
    data = {
        'features': features,
        'keypoints_locations': keypoints_locations
    }
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

def load_features_and_keypoints(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data['features'],data['keypoints_locations']

# 保存特征和关键点的位置信息
save_features_and_positions(features,keypoints_locations, 'features_and_keypoints.pkl')

# 在以后的运行中加载特征和关键点的位置信息
features,keypoints_locations = load_features_and_keypoints('features_and_keypoints.pkl')

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

