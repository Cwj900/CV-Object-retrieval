import os
import numpy as np
from SearchEngine import BoVW, serach_engine

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

    def get_result_images(self, input_image_name, k):
        dataset_path = 'image_paths.csv'
        vocaluraly_path = r'10\10.npy'
        sifts_features_path = 'image_features_list.pkl'
        images_representation_path = r'10\images_representation.npy'
        idf_path = r'10\idf.npy'

        bovw = BoVW(vocaluraly_path, dataset_path, sifts_features_path)
        serach_eng = serach_engine(bovw, input_image_name, vocaluraly_path, images_representation_path, dataset_path, idf_path, k)
        serach_eng.build_inverted_index()
        input_image_BoVW = serach_eng.co_input_BoVW()
        relevant_image_indexes = serach_eng.co_image_index(input_image_BoVW)
        lists = serach_eng.search(relevant_image_indexes, input_image_BoVW)

        return lists

    def compute_precision_recall(self,retrieved_results,good_paths,ok_paths,junk_paths):
        positive = 0
        blank = 0
        negative = 0

        for image in retrieved_results:
            if image in good_paths or image in ok_paths:
                positive += 1
            elif image in junk_paths:
                blank += 1
            else:
                negative += 1

        precision = positive / len(retrieved_results)
        recall = positive / (len(good_paths) + len(ok_paths))

        return precision, recall

    def compute_average_precision(self,precisions, recalls):
        precisions = np.concatenate(([0.], precisions, [0.]))
        recalls = np.concatenate(([0.], recalls, [1.]))

        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        change_indices = np.where(recalls[1:] != recalls[:-1])[0]
        average_precision = np.sum((recalls[change_indices + 1] - recalls[change_indices]) * precisions[change_indices + 1])

        return average_precision

    def evaluate(self):
        APs = []
        mAPs = []
        for i in range(55):
            k_data=[10,20,30,40,50]
            precisions=[]
            recalls=[]
            for k in k_data:
                retrieved_results=self.get_result_images(self.test_image_paths[i],k)
                precision, recall=self.compute_precision_recall(retrieved_results, self.test_good_paths[i],self.test_ok_paths[i],self.test_junk_paths[i])
                precisions.append(precision)
                recalls.append(recall)
            AP=self.compute_average_precision(precisions, recalls)
            print('precisions:',precisions)
            print(' recalls:',recalls)
            print("AP:",AP)
            APs.append(AP)
            if i%5==4:
                mAPs.append(sum(APs) / len(APs))
                APs=[]
        
        return sum(mAPs) / len(mAPs)

if __name__ == "__main__":
    evaluation=Evaluation()
    evaluation.load_data()
    score=evaluation.evaluate()
    print("模型性能：",score)