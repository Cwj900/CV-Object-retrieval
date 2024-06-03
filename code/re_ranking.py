import cv2
import numpy as np
from tqdm import tqdm
import pickle


class ReRanking:
    def __init__(self,input_keypoint,input_des,retrieval_results,idx_list,keypoints_des_path,max_num_images=800,max_failed_attempts=20,max_iterations=1000,reprojection_threshold=20):
        '''
        input_keypoint:检索图像keypoint坐标
        input_des：检索图像sift特征

        keypoints_des_path：包含模型keypoint和sift特征的文件
        ——> self.descriptors：所有图像的sift特征
            self.keypoints：所有图像的keypoint坐标
        
        retrieval_results：初次检索的图像地址列表（取前800个）
        idx_list：初次检索的图像索引列表（取前800个）


        max_num_images：重排的图像数量（800）

        max_failed_attempts：最大变换失败次数
        max_iterations：一张图图片的RANSAC迭代次数
        reprojection_threshold：RANSAC确定为内点的阈值
        '''
        self.input_keypoint=input_keypoint
        self.input_des=input_des

        with open(keypoints_des_path, 'rb') as file:
            data = pickle.load(file)
        self.descriptors=data['features']
        self.keypoints=data['keypoints_locations']

        self.max_num_images=max_num_images if len(retrieval_results)>max_num_images else len(retrieval_results)
        self.retrieval_results=retrieval_results[:max_num_images]
        self.idx_list=idx_list[:self.max_num_images]

        self.max_failed_attempts=max_failed_attempts
        self.max_iterations=max_iterations
        self.reprojection_threshold=reprojection_threshold

    #检索所有图像得到所有图像变换后的内点数量，并按照数量排序
    def spatial_verification(self):
        verified_images = []
        failed_attempts = 0
        # 遍历前max_num_images个检索结果
        for i in tqdm(range(self.max_num_images),total=self.max_num_images):
            image=self.retrieval_results[i]
            img_idx=self.idx_list[i]
            keypoint=self.keypoints[img_idx]
            descriptor=self.descriptors[img_idx]
            
            # 使用RANSAC估计图像间变换
            max_num_inliers=self.estimate_affine_ransac(keypoint,descriptor,self.max_iterations,self.reprojection_threshold)
            
            # 评估变换质量
            if max_num_inliers>=4:
                verified_images.append((image,max_num_inliers))
                failed_attempts=0
            else:
                failed_attempts+=1
                
            # 判断连续失败尝试次数是否超过阈值
            if failed_attempts>=self.max_failed_attempts:
                break

        # 根据内点对数量对验证的图像进行排序
        verified_images.sort(key=lambda x:x[1],reverse=True)
        # 提取验证通过的图像
        ranked_results=[image for image,_ in verified_images]

        return ranked_results 
         
    #计算RANSAC后的最佳模型的内点对数量
    def estimate_affine_ransac(self,keypoint,des,max_iterations,reprojection_threshold):
        """
        使用RANSAC算法估计两幅图像间的仿射变换
        
        输入:
        keypoint:检索图像的keypoint
        des:检索图像的descriptor
        max_iterations：迭代次数
        reprojection_threshold：确定为内点的阈值

        输出:
        max_num_inliers:内点对数量
        """
        matcher=cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        matches=matcher.match(des, self.input_des)

        src_pts=np.float32([keypoint[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
        dst_pts=np.float32([self.input_keypoint[m.trainIdx] for m in matches]).reshape(-1, 1, 2)

        best_model=None
        max_num_inliers=0

        for _ in range(max_iterations):
            # 从匹配的关键点中随机选择四对点
            if len(matches)>=4:
                random_indices=np.random.choice(len(matches),5,replace=False)
                src_pts_random=src_pts[random_indices]
                dst_pts_random=dst_pts[random_indices]

                # 估计仿射变换矩阵
                model,mask=cv2.findHomography(src_pts_random,dst_pts_random,cv2.RANSAC,reprojection_threshold)

                if model is not None:
                    # 计算所有关键点的变换后位置
                    transformed_pts=cv2.perspectiveTransform(src_pts,model)
                    # 计算变换后位置与目标位置的距离
                    distances=np.linalg.norm(transformed_pts-dst_pts,axis=2)
                    # 根据阈值判断内点
                    inliers=np.where(distances<=reprojection_threshold)[0]
                    num_inliers=len(inliers)

                    # 更新最佳模型和内点数量
                    if num_inliers>max_num_inliers:
                        best_model=model
                        max_num_inliers=num_inliers

        return max_num_inliers

'''
keypoints_descriptor_path='dataset/features_and_keypoints.pkl'
re_ranking=ReRanking(input_keypoint=input_keypoints_locations,input_des=input_sifts,retrieval_results=lists,idx_list=idx_list,keypoints_des_path=keypoints_descriptor_path)
ranked_results=re_ranking.spatial_verification()
print(ranked_results)
'''
