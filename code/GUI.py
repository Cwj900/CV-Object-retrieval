import tkinter as tk
from PIL import Image, ImageTk
from SearchEngine import BoVW, serach_engine

class ImageSearchApp:
    def __init__(self, window):
        self.window = window
        self.canvas = None
        self.entry = None
        self.button = None
        self.scrollbar = None
        self.image_objects = []

        self.create_widgets()

    def create_widgets(self):
        # 添加一个输入框和按钮
        self.entry = tk.Entry(self.window)
        self.entry.grid(row=0, column=0)

        self.button = tk.Button(self.window, text="Search", command=self.show_result_images)
        self.button.grid(row=0, column=1)

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

    def get_result_images(self, input_image_name):

        dataset_path = 'image_paths.csv'
        vocaluraly_path = r'10\10.npy'
        sifts_features_path = 'image_features_list.pkl'
        images_representation_path = r'10\images_representation.npy'
        idf_path=r'10\idf.npy'

        bovw = BoVW(vocaluraly_path, dataset_path, sifts_features_path)
        #BoVW_model = bovw.build_image_representation()
        serach_eng = serach_engine(bovw, input_image_name, vocaluraly_path, images_representation_path, dataset_path, idf_path,20)
        serach_eng.build_inverted_index()
        input_image_BoVW = serach_eng.co_input_BoVW()
        relevant_image_indexes = serach_eng.co_image_index(input_image_BoVW)
        lists = serach_eng.search(relevant_image_indexes, input_image_BoVW)
        print(lists)

        return lists

    def show_result_images(self):
        # 获取输入的图片名称，调用相应的函数获取相似图片的名称列表
        input_image_name = self.entry.get()
        similar_image_names = self.get_result_images(input_image_name)

        # 清空Canvas中的内容
        self.canvas.delete("all")

        # 加载和显示原始图片
        original_image = Image.open(input_image_name)
        original_image =original_image.resize((400, 400)) # 调整图片大小
        original_image_tk = ImageTk.PhotoImage(original_image)

        # 在Canvas上放置原始图片和文本标签
        self.canvas.create_image(0, 0, anchor="nw", image=original_image_tk)
        self.canvas.create_text(200, 20, text="原始图片", fill="red", font=("Arial", 16), anchor="nw")

        # 保存图像对象的列表
        self.image_objects.append(original_image_tk)

        # 加载和显示相似图片
        for i, image_name in enumerate(similar_image_names):
            similar_image = Image.open(image_name)
            similar_image = similar_image.resize((400, 400)) # 调整相似图片的大小
            similar_image_tk = ImageTk.PhotoImage(similar_image)

            # 在Canvas上放置相似图片和文本标签
            self.canvas.create_image(0, (i + 1) * 400, anchor="nw", image=similar_image_tk)
            self.canvas.create_text(200, (i + 1) * 400 + 20, text=f"结果图片 {i+1}", fill="red", font=("Arial", 16), anchor="nw")

            # 保存图像对象的列表
            self.image_objects.append(similar_image_tk)

            # 更新Canvas的可滚动区域大小
            self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def run(self):
        # 运行窗口的主循环
        self.window.mainloop()

if __name__ == "__main__":
    window = tk.Tk()
    app = ImageSearchApp(window)
    app.run()