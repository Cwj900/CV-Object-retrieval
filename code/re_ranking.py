from SearchEngine import lists,input_sifts,input_keypoints_locations,idx_list
from skimage.measure import ransac
from skimage.transform import AffineTransform
import cv2
import numpy as np
from tqdm import tqdm
import pickle

#执行空间验证并重排序检索结果
def spatial_verification(input_keypoint,input_des,retrieval_results,idx_list,keypoints,descriptors,max_num_images=1000, max_failed_attempts=20):
    """   
    参数:
    
    retrieval_results (list): 初始检索结果的图像列表
    max_num_images (int): 最大验证图像数量
    max_failed_attempts (int): 最大连续失败尝试次数
    
    返回:
    重排序后的图像列表
    """
    verified_images = []
    failed_attempts = 0
    

    max_num_images=max_num_images if len(retrieval_results)>max_num_images else len(retrieval_results)
    retrieval_results=retrieval_results[:max_num_images]
    idx_list=idx_list[:max_num_images]

    # 遍历前 max_num_images个检索结果
    for i in tqdm(range(max_num_images),total=max_num_images):
        image=retrieval_results[i]
        img_idx=idx_list[i]
        keypoint=keypoints[img_idx]
        descriptor=descriptors[img_idx]
        
        # 使用RANSAC估计图像间变换
        max_num_inliers=estimate_affine_ransac(keypoint,descriptor,input_keypoint,input_des)
        
        # 评估变换质量
        if max_num_inliers>=4:
            verified_images.append((image,max_num_inliers))
            failed_attempts=0
        else:
            failed_attempts+=1
            
        # 判断连续失败尝试次数是否超过阈值
        if failed_attempts>=max_failed_attempts:
            break

    # 根据内点对数量对验证的图像进行排序
    verified_images.sort(key=lambda x:x[1],reverse=True)

    # 提取验证通过的图像
    ranked_results=[image for image,_ in verified_images]

    return ranked_results 
            

def estimate_affine_ransac(keypoint,des,input_keypoint,input_des,max_iterations=10000,reprojection_threshold=3):
    """
    使用RANSAC算法估计两幅图像间的仿射变换
    
    输入:
    keypoint:检索图像的keypoint
    des:检索图像的descriptor
    input_keypoint:输入图像的keypoint
    input_des:输入图像的descriptor
    
    返回:
    best_model:最佳仿射变换矩阵
    inliers:内点对列表
    """
    matcher=cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    matches=matcher.match(des, input_des)

    src_pts=np.float32([keypoint[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
    dst_pts=np.float32([input_keypoint[m.trainIdx] for m in matches]).reshape(-1, 1, 2)

    best_model=None
    max_num_inliers=0

    for _ in range(max_iterations):
        # 从匹配的关键点中随机选择三对点
        if len(matches)>=4:
            random_indices=np.random.choice(len(matches),4,replace=False)
            src_pts_random=src_pts[random_indices]
            dst_pts_random=dst_pts[random_indices]

            # 估计仿射变换矩阵
            model,mask=cv2.findHomography(src_pts_random,dst_pts_random,cv2.RANSAC, reprojection_threshold)

            if model is not None:
                # 计算所有关键点的变换后位置
                transformed_pts=cv2.perspectiveTransform(src_pts,model)
                # 计算变换后位置与目标位置的距离
                distances=np.linalg.norm(transformed_pts - dst_pts, axis=2)
                # 根据阈值判断内点
                inliers=np.where(distances<=reprojection_threshold)[0]
                num_inliers=len(inliers)

                # 更新最佳模型和内点数量
                if num_inliers>max_num_inliers:
                    best_model=model
                    max_num_inliers=num_inliers

    return max_num_inliers


keypoints_location_path='dataset/features_and_keylocation.pkl'

with open(keypoints_location_path, 'rb') as file:
        data = pickle.load(file)
descriptors=data['features']
keypoints=data['keypoints_locations']

print(len(descriptors))
print(len(keypoints))
print(len(input_sifts),len(input_keypoints_locations))
ranked_results=spatial_verification(input_keypoints_locations,input_sifts,lists,idx_list,keypoints,descriptors,max_num_images=1000, max_failed_attempts=20)
print(ranked_results)

