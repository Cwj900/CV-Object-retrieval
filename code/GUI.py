import tkinter as tk
from PIL import Image, ImageTk
from Search import BoVW, serach_engine
import os
from tkinter import filedialog
from re_ranking import ReRanking

class ImageSearchApp:
    def __init__(self, window):
        self.window = window
        self.canvas = None
        self.button = None
        self.scrollbar = None
        self.image_path = None
        self.image_objects = []

        self.create_widgets()

    def create_widgets(self):

        # 添加一个按钮，用于打开本地图片文件夹
        self.button = tk.Button(self.window, text="Search", command=self.open_folder)
        self.button.grid(row=0, column=0)

        # 创建一个Canvas用于显示图片并添加垂直滚动条
        self.canvas = tk.Canvas(self.window)
        self.canvas.grid(row=1, column=0, columnspan=2, rowspan=2, sticky="nsew", padx=10, pady=10)

        self.scrollbar = tk.Scrollbar(self.window, command=self.canvas.yview)
        self.scrollbar.grid(row=1, column=2, rowspan=2, sticky="ns", padx=10)

        # 设置行和列的权重
        self.window.rowconfigure(1, weight=1)  # 图像显示区域占据更多的垂直空间
        self.window.columnconfigure(0, weight=1)  # 图像显示区域占据更多的水平空间

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))


    def open_folder(self):
        # 指定文件夹路径
        folder_path = "oxbuild_images"

        if os.path.exists(folder_path):
            # 获取文件夹路径的绝对路径
            folder_abs_path = os.path.abspath(folder_path)

            # 打开文件选择对话框，选择图片文件
            image_file = filedialog.askopenfilename(initialdir=folder_abs_path, filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))

            if image_file:
                # 获取图片文件的绝对路径
                image_abs_path = os.path.abspath(image_file)

                # 获取图片文件相对于文件夹路径的相对路径
                relative_path = os.path.relpath(image_abs_path, folder_abs_path)
                self.image_path="oxbuild_images\\"+relative_path
                # 显示结果图片
                self.show_result_images()
        else:
            print("文件夹不存在。")
    
    def get_result_images(self, input_image_name):

        dataset_path = 'image_paths.csv'
        vocaluraly_path = r'10000\10000.npy'
        sifts_features_path = 'features_and_keypoints.pkl'
        images_representation_path = r'10000\images_representation.npy'
        idf_path=r'10000\idf.npy'

        bovw = BoVW(vocaluraly_path, dataset_path, sifts_features_path)
        #BoVW_model = bovw.build_image_representation()
        serach_eng = serach_engine(bovw, input_image_name, vocaluraly_path, images_representation_path, dataset_path, idf_path)
        serach_eng.build_inverted_index()
        input_image_BoVW,input_sifts,input_keypoints_locations=serach_eng.co_input_BoVW()
        relevant_image_indexes = serach_eng.co_image_index(input_image_BoVW)
        lists,idx_list = serach_eng.search(relevant_image_indexes, input_image_BoVW)
        print(lists[:10])

        #re_ranking=ReRanking(input_keypoint=input_keypoints_locations,input_des=input_sifts,retrieval_results=lists,idx_list=idx_list,keypoints_des_path=sifts_features_path)
        #ranked_results=re_ranking.spatial_verification()
        #print(ranked_results[:10])

        return lists[:10] #,ranked_results[:10]
    
    def show_result_images(self):
        # 获取输入的图片名称，调用相应的函数获取相似图片的名称列表
        input_image_name = self.image_path
        similar_image_names = self.get_result_images(input_image_name)

        # 清空Canvas中的内容
        self.canvas.delete("all")

        # 加载和显示原始图片
        original_image = Image.open(input_image_name)
        original_image =original_image.resize((160, 160)) # 调整图片大小
        original_image_tk = ImageTk.PhotoImage(original_image)

        # 在Canvas上放置原始图片和文本标签
        self.canvas.create_image(100, 0, anchor="nw", image=original_image_tk)


        # 保存图像对象的列表
        self.image_objects.append(original_image_tk)

        # 加载和显示相似图片
        for i, image_name in enumerate(similar_image_names):
            similar_image = Image.open(image_name)
            similar_image = similar_image.resize((70, 70)) # 调整相似图片的大小
            similar_image_tk = ImageTk.PhotoImage(similar_image)

            # 计算图片在Canvas中的行和列索引
            row = i // 5
            column = i % 5

            # 在Canvas上放置相似图片
            self.canvas.create_image(column * 70, row * 70+170, anchor="nw", image=similar_image_tk)        

            # 保存图像对象的列表
            self.image_objects.append(similar_image_tk)
            '''
        # 加载和显示重排后的结果
        for i, image_name in enumerate(ranked_results):
            similar_image = Image.open(image_name)
            similar_image = similar_image.resize((70, 70)) # 调整相似图片的大小
            similar_image_tk = ImageTk.PhotoImage(similar_image)

            # 计算图片在Canvas中的行和列索引
            row = i // 5
            column = i % 5

            # 在Canvas上放置相似图片
            self.canvas.create_image(column * 70, row * 70+320, anchor="nw", image=similar_image_tk)        

            # 保存图像对象的列表
            self.image_objects.append(similar_image_tk)
            '''
            # 更新Canvas的可滚动区域大小
            self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def run(self):
        # 运行窗口的主循环
        self.window.mainloop()

if __name__ == "__main__":
    window = tk.Tk()
    app = ImageSearchApp(window)
    app.run()