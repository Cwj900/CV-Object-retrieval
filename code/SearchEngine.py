import numpy as np
import math
import pandas as pd
import pickle
from tqdm import tqdm
import cv2
from numpy.linalg import norm

class BoVW:
    def __init__(self,vocabulary_path,dataset_path,sifts_features_path):
        self.vocabulary=np.load(vocabulary_path,'r')
        self.num_vocabulary=len(self.vocabulary)

        dataset=[]
        data=pd.read_csv(dataset_path)
        list=data.values.tolist()
        for i in range(len(list)):
            dataset.append(list[i][0])

        self.N=len(dataset)
        with open(sifts_features_path, 'rb') as file:
            self.sifts_features = pickle.load(file)

        self.idf=[]

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
            tf:一个图像的各个vocabulary出现频率统计[len(vocabulary),]
        '''
        image_vocabulary_fre=np.zeros(self.num_vocabulary)
        for sift in sifts:
            #得到这个sift最近的vocabulary的idx
            nearest_word=self.fine_nearest_word(sift)
            #计算频数
            image_vocabulary_fre[nearest_word]+=1
        n=np.sum(image_vocabulary_fre)
        tf=[word/n for word in image_vocabulary_fre]
        return tf

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
        print('计算所有图像的各个视觉词汇的出现频率...')
        images_tf=[]
        for sifts in tqdm(self.sifts_features):
            tf=self.build_vocabulary_fre(sifts) #tf:len(vocabulary)
            images_tf.append(tf)#[num_image,len(vocabulary)]
        
        #计算idf
        df_word=np.zeros(self.num_vocabulary)
        #对每个图象的tf
        print('计算所有idf...')
        for tf in images_tf:
            for i,word in enumerate(tf):
                if word!=0:
                    df_word[i]+=1
        self.idf=[math.log(self.N/(word+1)) for word in df_word]#+1为了防止分母为0

        #计算所有图片的表示向量
        for tf in images_tf:
            image_representation=[tf[i]*self.idf[i] for i in range(self.num_vocabulary)]
            images_representation.append(image_representation)
        np.save('dataset/images_representation.npy',images_representation)
        np.save('dataset/idf.npy',self.idf)
        return images_representation



class serach_engine:
    def __init__(self,BoVW_model,input_image,vocabulary_path,images_representation_path,dataset_path,idf_path,k):
        self.BoVW_model=BoVW_model
        self.input_image=input_image
        self.vocabulary=np.load(vocabulary_path,'r')
        self.images_representation=np.load(images_representation_path,'r')
        self.idf=np.load(idf_path,'r')
        self.k=k

        self.IVF=[[]]*len(self.vocabulary)     
        self.dataset=[]
        data=pd.read_csv(dataset_path)
        list=data.values.tolist()
        for i in range(len(list)):
            self.dataset.append(list[i][0]) 

    #构建倒排索引
    def build_inverted_index(self):
        '''
        输入：
            images_representation：采用td-idf权重更新后的所有图片的向量
        输出：
            indexes：反向索引列表
        '''
        print('计算反向索引...')
        for i,image in enumerate(self.images_representation):
            #对每张图片
            for j,word in enumerate(image):
                if word!=0:
                    self.IVF[j]=self.IVF[j]+[i]
        #储存倒排索引
        with open('IVF.pkl', 'wb') as file:
            pickle.dump(self.IVF, file)
        return self.IVF

    #计算输入图像的BoVW特征向量
    def co_sift(self,image):
        image = cv2.imread(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
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
        image_BoVW=[tf[i]*self.idf[i] for i in range(len(self.vocabulary))]
        return image_BoVW
    
    #查找倒排索引
    def co_image_index(self,input_image_BoVW):
        with open('IVF.pkl', 'rb') as file:
            self.IVF = pickle.load(file)
        #初始化具有相同特征的图像索引列表
        relevant_image_indexes=[]
        for i,word in enumerate(input_image_BoVW):
            if word!=0:
                relevant_image_indexes+=self.IVF[i]
        relevant_image_indexes=list(set(relevant_image_indexes))
        return relevant_image_indexes

    #找出最接近的k个图像
    def cosine_similarity(self,vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = norm(vector1)
        norm_vector2 = norm(vector2)
        similarity = dot_product / (norm_vector1 * norm_vector2)
        return similarity
    def search(self,relevant_image_indexes,input_image_BoVW):
        search_image={}
        for image in relevant_image_indexes:
            #image的向量
            image_BoVW=self.images_representation[image]
            similarity=self.cosine_similarity(image_BoVW,input_image_BoVW)
            # distance=np.linalg.norm(image_BoVW-input_image_BoVW)
            search_image[image]=similarity
        #对结果进行排序
        # search_image=sorted(search_image.items(),key=lambda x:x[1])
        similar_image=sorted(search_image,key=search_image.get,reverse=True)
        similar_image=similar_image[1:self.k+1]#排除输入图片本身
        similar_image_path=[self.dataset[index] for index in similar_image]
        return similar_image_path


dataset_path='dataset/image_paths.csv'
vocaluraly_path='dataset/visual_words.npy'
sifts_features_path='dataset/image_features_list.pkl'
idf_path='dataset/idf.npy'
images_representation_path='dataset/images_representation.npy'
input_image='dataset\christ_church_001044.jpg'


bovw=BoVW(vocaluraly_path,dataset_path,sifts_features_path)
# BoVW_model=bovw.build_image_representation()
serach_eng=serach_engine(bovw,input_image,vocaluraly_path,images_representation_path,dataset_path,idf_path,20)
# serach_eng.build_inverted_index() 
input_image_BoVW=serach_eng.co_input_BoVW()
relevant_image_indexes=serach_eng.co_image_index(input_image_BoVW)
lists=serach_eng.search(relevant_image_indexes,input_image_BoVW)
print(lists)
