import os
import numpy as np
from tqdm import tqdm
from Search import BoVW, serach_engine

class Evaluation:
    def __init__(self, data_dir="gt_files_170407"):
        self.data_dir = data_dir
        self.test_image_paths = []
        self.test_good_paths = []
        self.test_ok_paths = []
        self.test_junk_paths = []
        self.test_areas = []
        self.load_data()

    def load_data(self):
        for file_dir in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file_dir)
            with open(file_path, 'r') as file:
                data = file.read()
                if file_dir.endswith('.txt') and 'good' in file_dir:
                    data_list = data.split('\n')
                    image_paths = []
                    for image_name in data_list[:-1]:
                        image_path = os.path.join("oxbuild_images", image_name + ".jpg")
                        image_paths.append(image_path)
                    self.test_good_paths.append(image_paths)
                elif file_dir.endswith('.txt') and 'ok' in file_dir:
                    data_list = data.split('\n')
                    image_paths = []
                    for image_name in data_list[:-1]:
                        image_path = os.path.join("oxbuild_images", image_name + ".jpg")
                        image_paths.append(image_path)
                    self.test_ok_paths.append(image_paths)
                elif file_dir.endswith('.txt') and 'junk' in file_dir:
                    data_list = data.split('\n')
                    image_paths = []
                    for image_name in data_list[:-1]:
                        image_path = os.path.join("oxbuild_images", image_name + ".jpg")
                        image_paths.append(image_path)
                    self.test_junk_paths.append(image_paths)
                elif file_dir.endswith('.txt') and 'query' in file_dir:
                    split_data = data.split(" ")
                    image_name = split_data[0].replace("oxc1_", "")
                    image_path = os.path.join("oxbuild_images", image_name + ".jpg")
                    x1 = float(split_data[1])
                    y1 = float(split_data[2])
                    x2 = float(split_data[3])
                    y2 = float(split_data[4])
                    area = [x1, y1, x2, y2]
                    self.test_image_paths.append(image_path)
                    self.test_areas.append(area)

    def get_result_images(self, input_image_name, k,test_area):
        dataset_path = 'image_paths.csv'
        vocaluraly_path = r'10000\10000.npy'
        sifts_features_path = 'features_and_keypoints.pkl'
        images_representation_path = r'10000\images_representation.npy'
        idf_path = r'10000\idf.npy'

        bovw = BoVW(vocaluraly_path, dataset_path, sifts_features_path)
        #BoVW_model=bovw.build_image_representation()
        serach_eng = serach_engine(bovw, input_image_name, vocaluraly_path, images_representation_path, dataset_path, idf_path,test_area)
        serach_eng.build_inverted_index()
        input_image_BoVW,input_sifts,input_keypoints_locations=serach_eng.co_input_BoVW()
        relevant_image_indexes = serach_eng.co_image_index(input_image_BoVW)
        lists,idx_list = serach_eng.search(relevant_image_indexes, input_image_BoVW)

        return lists[0:k]

    def compute_average_precision(self, l, results):
        pos=self.test_good_paths[l]+self.test_ok_paths[l]
        amb=self.test_junk_paths[l]

        old_recall = 0.0
        old_precision = 1.0
        ap = 0.0

        intersect_size = 0
        m = 0
        n = 0
        while m < len(results):
            if results[m] in amb:
                m += 1
                continue
            if results[m] in pos:
                intersect_size += 1
        
            recall = intersect_size / len(pos)
            precision = intersect_size / (n + 1)

            ap += (recall - old_recall) * ((old_precision + precision) / 2)

            old_recall = recall
            old_precision = precision
            m += 1
            n += 1

        return ap
 

    def evaluate(self):
        APs = []
        aps=[]
        mAPs = []
        for i in tqdm(range(55)):
            if len(self.test_good_paths[i])+len(self.test_ok_paths[i])>50:
                k=len(self.test_good_paths[i])+len(self.test_ok_paths[i])
            else:
                k=50

            retrieved_results=self.get_result_images(self.test_image_paths[i],k,self.test_areas[i])

            AP=self.compute_average_precision(i,retrieved_results)
            APs.append(AP)
            if i%5==4:
                mAPs.append(sum(APs) / len(APs))
                aps=aps+APs
                APs=[]
        print(aps)
        return sum(mAPs) / len(mAPs)

if __name__ == "__main__":
    evaluation=Evaluation()
    evaluation.load_data()
    score=evaluation.evaluate()
    print("模型性能：",score)