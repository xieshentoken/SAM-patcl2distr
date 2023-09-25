import cv2
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import os
import tkinter as tk
from collections import OrderedDict
from itertools import permutations
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk

import patcl2distr_library as p2d

class PATCL2DistrGUI:
    def __init__(self, master):
        self.master = master
        self.initWidgets()

        self.statistics = None
        self.workspace = os.path.dirname(os.path.abspath(__file__))
        self.scalebar_module = self.workspace+'/image/scalebar module/scalebar1.jpg'
        self.scalebar_image, self.pixel_distance, self.physic_distance = None, None, None
        self.weight = "/Users/helong/Documents/CV practise/sam_vit_h_4b8939.pth"
        self.device = 'cpu'
        self.model_type = 'default'
        self.image_file_path = None
        self.result = None

    def initWidgets(self):
        self.master.title("patcl2distr")
        self.master.geometry("700x700")

        # 初始化菜单、工具条用到的图标
        self.init_icons()
        # 调用init_menu初始化菜单
        self.init_menu()

        # Frame1: 选择图片文件
        frame1 = ttk.Frame(self.master, padding=(10, 10))
        frame1.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.NO)

        ttk.Label(frame1, text='Image Path:', font=('StSong', 20, 'bold')
                                         ).pack(side=tk.LEFT, ipadx=5, ipady=5, padx= 10)
            # 创建字符串变量，用于传递待测图片地址
        self.image_file_adr = tk.StringVar()
        self.image_file_adr.set('Select your SEM image here.')
            # 创建Entry组件，将其textvariable绑定到self.image_file_path变量
        ttk.Entry(frame1, textvariable=self.image_file_adr,
            width=28,
            font=('StSong', 20, 'bold'),
            foreground='#8080c0').pack(side=tk.LEFT, ipadx=5, ipady=5)
        openfile_s = ttk.Button(frame1, text='select image', width = 13,
            command=self.load_image                                                # 绑定load_image方法
            )
        openfile_s.pack(side=tk.LEFT, ipadx=1, ipady=5)

        # Frame2: 处理文件按钮
        frame2 = ttk.Frame(self.master, padding=(10, 10))
        frame2.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.NO)
        scalebar_config = ttk.Button(frame2, text='Config ScaleBar', width = 13,
            command=self.scale_bar_config                                          # 配置scalebar
            )
        scalebar_config.pack(side=tk.LEFT, ipadx=1, ipady=5)
        scalebar_config.bind("<Button-3>", self.scalebar_info_to_xls)              # 绑定右键单击将标尺识别的信息保存至excel
        process_image = ttk.Button(frame2, text='Process image', width = 13,
            command=self.process_image                                             # 进行图像分割并显示统计结果
            )
        process_image.pack(side=tk.LEFT, ipadx=1, ipady=5)
        process_image.bind("<Button-3>", self.save_statistics_csv)                # 绑定右键单击将统计结果保存至新建的以图片名命名的文件夹中
        new_image = ttk.Button(frame2, text='Clear image', width = 13,
            command=self.new_image                                                # 不改变标尺信息，重新选择要处理的图片
            )
        new_image.pack(side=tk.LEFT, ipadx=1, ipady=5)

        # Frame3: 画布
        frame3 = ttk.Labelframe(self.master, text='ScaleBar                                                                                Segment result',
                                 padding=(10, 10))
        frame3.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.NO)
        self.canvas1 = tk.Canvas(frame3, width=320, height=240, bg="white")
        self.canvas1.pack(side=tk.LEFT, padx=(10, 5), pady=10)

        self.canvas2 = tk.Canvas(frame3, width=320, height=240, bg="white")
        self.canvas2.pack(side=tk.LEFT, padx=5, pady=10)
        # Frame4: 画布
        frame4 = ttk.Labelframe(self.master, text='Statistic Result', padding=(10, 10))
        frame4.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.NO)
        self.canvas3 = tk.Canvas(frame4, width=660, height=200, bg="white")
        self.canvas3.pack(side=tk.LEFT, padx=(5, 10), pady=10)

    # 创建menubar
    def init_menu(self):
        # '初始化菜单的方法'
        # 定义菜单条
        menus = ('File', 'Config', 'Process', 'Help')
        # 定义菜单数据
        items = (OrderedDict([
                # 每项对应一个菜单项，后面元组第一个元素是菜单图标，
                # 第二个元素是菜单对应的事件处理函数
                ('New', (None, None)),
                ('Open', (None, None)),
                ('-1', (None, None)),
                ('Set work space', (None, self.set_workspace)),
                ('Save CSV', (None, self.save_statistics_csv)),
                ('-2', (None, None)),
                ('Exit', (None, None)),
                ]),
            OrderedDict([('Scale Bar',OrderedDict([('ScaleBar module',(None, None)),
                ('Config  file path',(None, None)),
                ('-1',(None, None)),
                ('ScaleBar info Save',(None, self.scalebar_info_to_xls))])), 
                ('-1', (None, None)),
                ('segment-anything', (None, self.config_sam)),
                ]),
            OrderedDict([('Show masks',(None, self.showMasks)),
                         ('Show Segments',(None, self.showSegments)),
                         ('Show Statistics',(None, self.showStatistics)),
                         ('-1',(None, None)), 
                         ('Packing Optimization',(None, None)),
                         ('Merge csv',(None, self.merge_two_csv)),
                         ('-2',(None, None)), 
                         ('Statistics from csv',(None, self.plot_bar_from_csv)), 
                ]),
            OrderedDict([('Help',(None, None)), 
                ('-1',(None, None)),
                ('About', (None, self.show_about))]))
        # 使用Menu创建菜单条
        menubar = tk.Menu(self.master)
        # 为窗口配置菜单条，也就是添加菜单条
        self.master['menu'] = menubar
        # 遍历menus元组
        for i, m_title in enumerate(menus):
            # 创建菜单
            m = tk.Menu(menubar, tearoff=0)
            # 添加菜单
            menubar.add_cascade(label=m_title, menu=m)
            # 将当前正在处理的菜单数据赋值给tm
            tm = items[i]
            # 遍历OrderedDict,默认只遍历它的key
            for label in tm:
                # print(label)
                # 如果value又是OrderedDict，说明是二级菜单
                if isinstance(tm[label], OrderedDict):
                    # 创建子菜单、并添加子菜单
                    sm = tk.Menu(m, tearoff=0)
                    m.add_cascade(label=label, menu=sm)
                    sub_dict = tm[label]
                    # 再次遍历子菜单对应的OrderedDict，默认只遍历它的key
                    for sub_label in sub_dict:
                        if sub_label.startswith('-'):
                            # 添加分隔条
                            sm.add_separator()
                        else:
                            # 添加菜单项
                            sm.add_command(label=sub_label,image=None,
                                command=sub_dict[sub_label][1], compound=tk.LEFT)
                elif label.startswith('-'):
                    # 添加分隔条
                    m.add_separator()
                else:
                    # 添加菜单项
                    m.add_command(label=label,image=None,
                        command=tm[label][1], compound=tk.LEFT)
    # 生成所有需要的图标
    def init_icons(self):
        pass

    def new_image(self):
        for widget in self.canvas2.winfo_children():
            widget.destroy()
        for widget in self.canvas3.winfo_children():
            widget.destroy()
        self.image_file_path = None
        self.result = None

    # 加载图片
    def load_image(self):
        self.image_file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if self.image_file_path:
            self.image_file_adr.set(self.image_file_path)
        try:
            with open(self.workspace + '/config/'+self.image_file_path.split('/')[-1].split('.')[0]+'_scalebar_cofig.txt', 'r') as f:
                lines = f.readlines()
            self.pixel_distance = int(float(lines[2].strip().split('\n')[0]))
            self.physic_distance = int(float(lines[3].strip().split('\n')[0]))
        except:
            messagebox.showinfo(title='警告',message='配置标尺信息')
        print('pixel and scalebar:', self.pixel_distance, self.physic_distance)

    # 读取比例尺信息
    def scale_bar_config(self):
        scale_config_filename = self.workspace +'/config/'+ self.image_file_path.split('/')[-1].split('.')[0] +'_scalebar_cofig.txt'
        self.scalebar_image, self.pixel_distance, self.physic_distance = p2d.scalebar_config(
            self.image_file_path, scale_config_filename, self.scalebar_module)
        pil_image = Image.fromarray(self.scalebar_image).resize((320,240))
        self.image = ImageTk.PhotoImage(pil_image)
        self.canvas1.create_image(1, 1, anchor=tk.NW, image=self.image)
    
    def scalebar_info_to_xls(self,even=None):
        folder_path = self.workspace + r'/export data/' + self.image_file_path.split('/')[-1].split('.')[0]
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"文件夹 '{folder_path}' 不存在，已成功创建。")
        try:
            save_path = folder_path + r'/' + self.image_file_path.split('/')[-1].split('.')[0] + '_scalebar.xlsx'
            p2d.scalebar_info_to_xls(self.physic_distance, self.scalebar_image, save_path)
            messagebox.showinfo(title='警告',message='保存成功')
        except:
            messagebox.showinfo(title='警告',message='保存错误，请检查是否进行过标尺识别')
            pass

    def config_sam(self):
        sscc = SAM_Config(self.master, self.weight, self.device, self.model_type)
        self.weight, self.device, self.model_type = sscc.weight, sscc.device, sscc.model_type

    # 保存分割结果
    def save_statistics_csv(self, even=None):
        if self.result is not None:
            folder_path = self.workspace + r'/export data/' + self.image_file_path.split('/')[-1].split('.')[0]
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"文件夹 '{folder_path}' 不存在，已成功创建。")
            file_name = self.image_file_path.split('/')[-1].split('.')[0]
            self.result[0].to_csv(folder_path+r'/'+file_name+'_particle.csv', index=False)            # 粒径分布结果
            self.result[1].to_csv(folder_path+r'/'+file_name+'_statistics.csv', index=True)      # 统计结果
            messagebox.showinfo(title='警告',message='保存成功')
        else:
            messagebox.showinfo(title='警告',message='保存错误，请检查是否是否运行过分割功能')
            pass
    
    def process_image(self):
        self.masks = p2d.segment_anything_process(self.image_file_path, self.weight, self.device, self.model_type)
        self.result = p2d.segments_statistice(self.masks, self.pixel_distance, self.physic_distance)
        # 显示mask的分割结果
        fig = p2d.show_segments(self.image_file_path, self.masks)
        fig.set_size_inches(self.canvas2.winfo_width()/fig.dpi, self.canvas2.winfo_height()/fig.dpi)
        canvas2 = FigureCanvasTkAgg(fig, master=self.canvas2)
        canvas2.get_tk_widget().pack()
        # 显示粒径统计结果
        fig_stat = p2d.show_result(self.result[0], self.result[1], 100)
        fig_stat.set_size_inches(self.canvas3.winfo_width()/fig_stat.dpi, 1.5*self.canvas3.winfo_height()/fig_stat.dpi)
        canvas3 = FigureCanvasTkAgg(fig_stat, master=self.canvas3)
        canvas3.get_tk_widget().pack()
        plt.close('all')

    def showMasks(self):
        if self.masks:
            p2d.show_mask(self.masks, 3)
        else:
            messagebox.showinfo(title='警告',message='无数据')
    def showSegments(self):
        if self.masks:
            p2d.show_segments(self.image_file_path, self.masks)
            plt.show()
        else:
            messagebox.showinfo(title='警告',message='无数据')
    def showStatistics(self):
        if self.masks:
            p2d.show_result(self.result[0], self.result[1], 100)
            plt.show()
        else:
            messagebox.showinfo(title='警告',message='无数据')
    def set_workspace(self):
        self.workspace = filedialog.askdirectory(initialdir=r'/Users/helong/code learning/packing')
    def plot_bar_from_csv(self):
        csv_path = filedialog.askopenfilename(initialdir=self.workspace + r'/export data', filetypes=[("CSV files", "*.csv")])
        p2d.plot_histogram_from_csv(csv_path, 100)
    # 把两个包含粒径统计信息的csv文件合并成一个，并生成一个包含新统计数据的csv，都放置在工作目录文件夹下
    def merge_two_csv(self):
        self.csv1_path, self.csv2_path = None, None
        popup = tk.Toplevel(self.master)
        popup.title("选择CSV文件")
        def open_popup():

            self.label_file1 = tk.Label(popup, text="第一个CSV文件路径:")
            self.label_file1.pack(pady=10)
            button_select_file1 = tk.Button(popup, text="选择文件", command=select_file1)
            button_select_file1.pack(pady=5)

            self.label_file2 = tk.Label(popup, text="第二个CSV文件路径:")
            self.label_file2.pack(pady=10)
            button_select_file2 = tk.Button(popup, text="选择文件", command=select_file2)
            button_select_file2.pack(pady=5)
            button_ok = tk.Button(popup, text="OK", command=ok_button)
            button_ok.pack(pady=20)

        def select_file1():
            file1 = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if file1:
                self.label_file1.config(text=file1)
                self.csv1_path = file1

        def select_file2():
            file2 = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if file2:
                self.label_file2.config(text=file2)
                self.csv2_path = file2

        def ok_button():
            popup.destroy()
            csv1 = pd.read_csv(self.csv1_path)
            csv2 = pd.read_csv(self.csv2_path)
            merged_csv = pd.concat([csv1, csv2], ignore_index=True)
            merged_csv.to_csv(self.workspace+'/'+
                            self.csv1_path.split('/')[-1].split('.')[0]+'_'+
                            self.csv2_path.split('/')[-1].split('.')[0]+'_merged.csv', 
                            index=False)
            merged_stati = round(pd.concat([merged_csv.mean(), merged_csv.std(), merged_csv.median(), 
                                            merged_csv.quantile(0.9), merged_csv.quantile(0.99)], axis=1).T, 1)
            merged_stati.index = ['average', 'standardization', 'D50', 'D90', 'D99']
            merged_stati.to_csv(self.workspace+'/'+
                            self.csv1_path.split('/')[-1].split('.')[0]+'_'+
                            self.csv2_path.split('/')[-1].split('.')[0]+'_mergedStati.csv', 
                            index=True)
            print(self.csv1_path, self.csv2_path)

        open_popup()
        

    def show_about(self):
        tk.messagebox.showinfo("关于", "这是一个用于处理图像的软件")

# 创建弹窗-----------------------------------------------------------------------------------------------------------
class SAM_Config(tk.Toplevel):
    # 定义构造方法
    def __init__(self, parent, weight, device, model_type, rgb='#8080c0', title = 'Config Segemnt-anything', modal=False):
        tk.Toplevel.__init__(self, parent)
        self.transient(parent)
        # 设置标题
        if title: self.title(title)
        self.parent = parent
        self.weight = weight
        self.device = device
        self.model_type = model_type
        self.rgb = rgb
        # 创建对话框的主体内容
        frame = tk.Frame(self)
        # 调用init_widgets方法来初始化对话框界面
        self.initial_focus = self.init_widgets(frame)
        frame.pack(padx=5, pady=5)
        # 根据modal选项设置是否为模式对话框
        if modal: self.grab_set()
        if not self.initial_focus:
            self.initial_focus = self
        # 为"WM_DELETE_WINDOW"协议使用self.cancel_click事件处理方法
        self.protocol("WM_DELETE_WINDOW", self.cancel_click)
        # 根据父窗口来设置对话框的位置
        self.geometry("+%d+%d" % (parent.winfo_rootx()+50,
            parent.winfo_rooty()+50))
        print( self.initial_focus)
        # 让对话框获取焦点
        self.initial_focus.focus_set()
        self.wait_window(self)

    def init_var(self):
        self.weight_v = tk.StringVar()
        self.device_v = tk.IntVar()
        self.model_type_v = tk.StringVar()

    # 通过该方法来创建自定义对话框的内容
    def init_widgets(self, master):
        self.init_var()
        
        # Frame1: 选择文件
        frame1 = ttk.Frame(master, padding=(10, 10))
        frame1.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.NO)

        ttk.Label(frame1, text='SAM checkpoint(weight):', font=('StSong', 20, 'bold')
                                         ).pack(side=tk.LEFT, ipadx=5, ipady=5, padx= 10)
        self.weight_v.set(self.weight)
        ttk.Entry(frame1, textvariable=self.weight_v,
            width=28,
            font=('StSong', 20, 'bold'),
            foreground='#8080c0').pack(side=tk.LEFT, ipadx=5, ipady=5)
        openfile_s = ttk.Button(frame1, text='select weight', width = 13,
            command=self.open_weight_file 
            )
        openfile_s.pack(side=tk.LEFT, ipadx=1, ipady=5)
        # Frame2: 选择加速类型
        frame2 = ttk.Frame(master, padding=(10, 10))
        frame2.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.NO)

        ttk.Label(frame2, text='Device:', font=('StSong', 20, 'bold')
                                         ).pack(side=tk.LEFT, ipadx=5, ipady=5, padx= 10)
            # 创建单选框
        self.device_v.set(0)
        radio1 = ttk.Radiobutton(frame2, text="cup", variable=self.device_v, value=0)
        radio1.pack(side=tk.LEFT, ipadx=5, ipady=5)
        radio2 = ttk.Radiobutton(frame2, text="cuda", variable=self.device_v, value=1)
        radio2.pack(side=tk.LEFT, ipadx=5, ipady=5)
        # Frame3: 选择model
        frame3 = ttk.Frame(master, padding=(10, 10))
        frame3.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.NO)

        ttk.Label(frame3, text='Model Type:', font=('StSong', 20, 'bold')
                                         ).pack(side=tk.LEFT, ipadx=5, ipady=5, padx= 10)
            # 创建下拉列表框
        self.model_type_v.set("default")
        combo = ttk.Combobox(frame3, textvariable=self.model_type_v, values=["default"])
        combo.pack(side=tk.LEFT, ipadx=5, ipady=5)
        # Frame4: OK
        frame4 = ttk.Frame(master, padding=(10, 10))
        frame4.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        ok_button = ttk.Button(frame4, text="OK", command=self.ok_button_click)
        ok_button.pack(side=tk.LEFT, ipadx=1, ipady=5)

    def open_weight_file(self):
        self.weight = filedialog.askopenfilename(filetypes=[("PTH files", "*.pth")])
        if self.weight:
            self.weight_v.set(self.weight)
            print(f"选择的权重文件路径为: {self.weight}")
    
        # 定义确定按钮的操作
    def ok_button_click(self):
        if self.device_v.get() == 0:
            self.device = 'cpu'
        elif self.device_v.get() == 1:
            self.device = 'cuda'
        self.model_type = self.model_type_v.get()
        print(self.weight, self.device, self.model_type)
        self.parent.focus_set()
        self.destroy()

    def cancel_click(self, event=None):
        # print('取消')
        # 将焦点返回给父窗口
        self.parent.focus_set()
        # 销毁自己
        self.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PATCL2DistrGUI(root)
    root.mainloop()
