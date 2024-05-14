import numpy as np
import math
import cv2
sift = cv2.SIFT_create()
class BoVW:
    def __init__(self,vocabulary,dataset,sifts_features_path):
        self.vocabulary=vocabulary
        self.num_vocabulary=len(vocabulary)
        self.N=len(dataset)
        self.sifts_features=np.load(sifts_features_path,'r')


    #找到sift最近的vocabulary的idx
    def fine_nearest_word(self,sift):
        '''
        计算与一个sift特征最接近的vocabulary的word的index
        输入：
            sift:一个图片的一个sift特征向量
            vocabulary：聚类后的视觉单词
        输出：
            nearest_word：离这个sift特征最近的视觉单词索引
        '''
        distance=np.linalg.norm(sift-self.vocabulary,axis=1)
        nearest_word=np.argmin(distance)
        return nearest_word


    #初始构建图像表示向量
    def build_vocabulary_fre(self,sifts):
        '''
        对一个图像统计vocabulary中每个word的出现频率，第一次得到图像的表示向量
        输入：
            sifts：一个图像的所有sift特征
            vocabulary：聚类后的视觉单词
        输出：
            ft:一个图像的各个vocabulary出现频率统计[len(vocabulary),]
        '''
        image_vocabulary_fre=np.zeros(self.num_vocabulary)
        for sift in sifts:
            #得到这个sift最近的vocabulary的idx
            nearest_word=self.fine_nearest_word(sift)
            #计算频数
            image_vocabulary_fre[nearest_word]+=1
        n=np.sum(image_vocabulary_fre)
        ft=[word/n for word in image_vocabulary_fre]
        return ft

    #使用TF-IDF权重
    def build_image_representation(self):
        '''
        输入：
            dataset:所有图片的路径列表
            vocabulary：聚类后的视觉单词
            images_feature:所有图像的sifts特征

        images_tf:存储所有图像的tf：[image_num,vocabulary_len]
        idf:储存所有vocabulary的idf：[len(vocabulary),]
        
        输出：
            images_representation：采用td-idf权重更新后的所有图片的向量[num_image,len(vocabulary)]

        '''
        #初始化保存所有图像向量的列表
        images_representation=[]

        #计算tf
        images_tf=[]
        for sifts in self.sifts_features:
            tf=self.build_vocabulary_fre(sifts) #tf:len(vocabulary)
            images_tf.append(tf)#[num_image,len(vocabulary)]
        
        #计算idf
        df_word=np.zeros(self.num_vocabulary)
        #对每个图象的tf
        for tf in images_tf:
            for i,word in enumerate(tf):
                if word!=0:
                    df_word[i]+=1
        self.idf=[math.log(self.N/(word+1)) for word in df_word]   #+1为了防止分母为0

        #计算所有图片的表示向量
        for tf in images_tf:
            image_representation=[tf[i]*self.idf[i] for i in range(self.num_vocabulary)]
            images_representation.append(image_representation)

        return images_representation



class serach_engine:
    def __init__(self,BoVW_model,input_image,vocabulary,k):
        self.BoVW_model=BoVW_model
        self.input_image=input_image
        self.vocabulary=vocabulary  
        self.k=k

        self.IVF=[]      

    #构建倒排索引
    def build_inverted_index(self,images_representation):
        '''
        输入：
            images_representation：采用td-idf权重更新后的所有图片的向量
        输出：
            indexes：反向索引列表
        '''
        for i,image in enumerate(images_representation):
            #对每张图片
            for j,word in enumerate(image):
                if word!=0:
                    self.IVF[j]+=[i]
        return self.IVF

    #计算输入图像的BoVW特征向量
    def co_sift(self,image):
       image = cv2.imread(image)
       gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       keypoints, descriptors = sift.detectAndCompute(gray_image, None)
       if descriptors is not None:
          return descriptors
       return None


    def co_input_BoVW(self):
        '''
        输入：输入图片的地址
        输出：输入图片的BoVW特征向量
        '''
        #提取sift特征向量
        sifts=self.co_sift(self.input_image)
        #获得所有vocabulary中所有word的出现频率
        tf=self.BoVW_model.build_vocabulary_fre(sifts)
        #计算td-idf权重更新后的所有图片的向量
        image_BoVW=[tf[i]*self.BoVW_model.idf[i] for i in range(len(self.vocabulary))]
        return image_BoVW
    
    #查找倒排索引
    def co_image_index(self,input_image_BoVW):
        relevant_image_indexes=[]
        for i,word in enumerate(input_image_BoVW):
            if word!=0:
                relevant_image_indexes+=self.IVF[i]
        return relevant_image_indexes

    #找出最接近的k个图像
    def search(self,relevant_image_indexes,images_representation,input_image_BoVW):
        search_image={}
        for image in relevant_image_indexes:
            #image的向量
            image_BoVW=images_representation[image]
            distance=np.linalg.norm(image_BoVW-input_image_BoVW)
            search_image[image]=distance
        #对结果进行排序
        search_image=sorted(search_image.items(),key=lambda x:x[1])
        similar_image=[key for key,value in search_image[:self.k]]
        return similar_image
