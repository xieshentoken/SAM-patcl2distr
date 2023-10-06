import cv2
import numpy as np
import pandas as pd
import pytesseract
import tkinter as tk
from tkinter import filedialog

import openpyxl
from openpyxl.drawing.image import Image
import io
import re

import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

# 读取包含标尺信息的图片，并通过单击选择标尺，获得像素与物理长度的关系
def scalebar_config(img_file, scalebar_cofig_file=r'/Users/helong/code learning/py/patcl2dist/config/scalebar_cofig.txt', match_template=r'/Users/helong/code learning/py/patcl2dist/image/scalebar1.jpg'):
    # 首先根据标尺图片template对目标图片进行模板匹配，寻找到标尺代表的物理距离-----------------
    template = cv2.imread(match_template)
    target = cv2.imread(img_file)
    result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    h,w = template.shape[:-1]
    cv2.rectangle(target, max_loc, (max_loc[0]+w, max_loc[1]+h), (0,0,255), 2)
    crop = target[max_loc[1]:max_loc[1]+h, max_loc[0]:max_loc[0]+w] 

    blank = np.zeros((10*h, 10*w, 3), np.uint8)
    x = (blank.shape[1] - w) // 2
    y = (blank.shape[0] - h) // 2
    blank[y:y+h, x:x+w] = crop

    text = pytesseract.image_to_string(blank)
    pattern = r'(\d+\.\d+)\s*(um|nm)'
    matches = re.findall(pattern, text)
    number, unit = 0, 'um'
    for match in matches:
        number, unit = match
    physic_distance = int(float(number))
    #-------------------------------------------------------------------------------
    # 鼠标单击选取标尺，读取像素值-------------------------------------------------------
    config_points = []
    def on_mouse_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONUP: # 左键点击
            if len(config_points) < 2:
                config_points.append((x,y))
                cv2.circle(image, (x,y), 10, (0, 255, 0), -1)
            else:
                print('config points is full')
    image = cv2.imread(img_file)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',on_mouse_event) # 设置鼠标事件

    while(True):
        # 显示配置点
        if len(config_points) == 2:
            cv2.line(image, config_points[0], config_points[1], (0, 0, 255), 5)

        cv2.imshow('image',image)
        # 退出ESC
        if cv2.waitKey(20) & 0xFF == 27:
            break
        # enter键保存配置文件
        elif cv2.waitKey(20) & 0xFF == 13:
            cv2.putText( image , 'Saved config', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            with open(scalebar_cofig_file, 'w', encoding='utf8') as fw:
                # 写入配置文件，每行一个点：x,y
                for pt in config_points:
                    write_line = str(pt[0]) + ',' + str(pt[1]) + '\n'
                    fw.write(write_line)
                # 计算点之间的像素距离
                pixel_distance = int(np.linalg.norm(np.array(config_points[0]) - np.array(config_points[1])))
                fw.write(str(pixel_distance)+ '\n')
                fw.write(str(physic_distance))
                print('pixel distance: {}\nphysic distance: {} {}'.format(pixel_distance, physic_distance, unit))

            print('save config file')

    cv2.destroyAllWindows()
    print(type(pixel_distance),type(physic_distance))
    return target, pixel_distance, physic_distance

# 保存标尺识别的结果至excel
def scalebar_info_to_xls(physic_distance, target, save_path):
    #图片尺寸
    IMAGE_SIZE = (320,240)

    # 创建新的Excel工作簿
    wb = openpyxl.Workbook()

    # 获取默认Sheet表格
    sheet = wb.active 

    # 在A1单元格插入字符串
    sheet['A1'] = physic_distance
    #修改图片大小
    resized = cv2.resize(target, IMAGE_SIZE)
    # 将图片数据转换为二进制格式
    is_success, img_buffer = cv2.imencode(".png", resized)
    io_img = io.BytesIO(img_buffer)
    # 在B1单元格插入图片 
    img = Image(io_img) 
    sheet.add_image(img, 'B1')

    # 设置A1和B1的列宽和行高
    sheet.row_dimensions[1].height = IMAGE_SIZE[1]*0.7458

    sheet.column_dimensions['B'].width = IMAGE_SIZE[0]*0.141
    sheet.row_dimensions[1].height = IMAGE_SIZE[1]*0.7458

    # 保存工作簿
    wb.save(save_path)

# 调用segment-anything模型进行图像分割
def segment_anything_process(img, sam_checkpoint=r"/Users/helong/Documents/CV practise/sam_vit_h_4b8939.pth", device="cpu", model_type="default"):
    # 配置权重文件，用cpu或gpu加速，以及分割模型
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # 选择要处理的图片
    image_file_name = img
    image = cv2.imread(image_file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    masks = mask_generator.generate(image)

    return masks

# 根据分割结果进行粒径统计
def segments_statistice(masks, pixel_distance, physic_distance):
    segments, areas = [], []
    contour_area, contour_len = [], []
    for mask in masks:
        segments.append(mask['segmentation'])
        mask_seg8 = mask['segmentation'].astype(np.uint8)
        contours, _ = cv2.findContours(mask_seg8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_area.append(cv2.contourArea(contours[0]))
        contour_len.append(cv2.arcLength(contours[0], True))
        areas.append(mask['area'])

    stati = np.concatenate([np.array(areas).reshape(len(masks),1), np.array(contour_area).reshape(len(masks),1), np.array(contour_len).reshape(len(masks),1)],axis=1)
    stati = pd.DataFrame(stati, columns=['area','cv2_area', 'cv2_length'])
    # 用segment-anything的model计算的面积进行换算，分别以圆形和正方形计算尺寸大小
    stati.insert(3, 'circle_dia/um', stati['area'].apply(lambda x: round(np.sqrt(x/np.pi)/pixel_distance*2*physic_distance,1)))
    stati.insert(4, 'square_dia/um', stati['area'].apply(lambda x: round(np.sqrt(x)/pixel_distance*physic_distance,1)))
    stati.drop([0,1], axis=0, inplace=True)        # 当图片中包含比例尺时执行此行，drop删除前两个mask，第一个mask是背景，第二个mask是比例尺
    # 统计均值、标准差、D50、D90、D99保存于stati_data中
    stati_data = round(pd.concat([stati.mean(), stati.std(), stati.median(), stati.quantile(0.9), stati.quantile(0.99)], axis=1).T, 1)
    stati_data.index = ['average', 'standardization', 'D50', 'D90', 'D99']
    return stati, stati_data

# 展示分割得到的mask
def show_mask(masks, rowcol=3):
    fig, axs = plt.subplots(rowcol, rowcol, figsize=(10,5))
    for i,mask in enumerate(masks[:rowcol**2]):
        mask_seg = mask['segmentation']
        mask_seg8 = mask_seg.astype(np.uint8)
        contours, _ = cv2.findContours(mask_seg8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area = cv2.contourArea(contours[0])
        ax = axs[i//rowcol, i%rowcol] 
        ax.imshow(mask_seg)
        ax.set_title('area:{0},segarea:{1},score:{2}'.format(area,round(mask['area']),round(mask['stability_score'],3)))
        
    plt.tight_layout()
    plt.show()

# 显示分割后的图
def show_segments(image_file_name, masks):
    image = cv2.imread(image_file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    def show_anns(anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)

    fig = plt.figure(figsize=(5,5))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    # plt.show() 
    return fig

# 绘制尺径分布图
def show_result(stati, stati_data, plot_label='circle_dia/um', bin_s=100):
    fig = plt.figure(figsize=(15, 3))
    hist, bins = np.histogram(stati[plot_label], bins=bin_s)
    plt.bar(bins[:-1], hist, width=np.diff(bins), ec='k')

    plt.title('mean:{}  std:{}  max:{}  D50:{}  D90:{}  D99:{}  unit:um'.format(stati_data[plot_label][0],stati_data[plot_label][1],stati[plot_label].max(),stati_data[plot_label][2],stati_data[plot_label][3],stati_data[plot_label][4])) 
    plt.grid(axis='y', alpha=0.3)
    plt.xlabel('particle size(um)')
    plt.ylabel('frequency')
    plt.tight_layout()
    # plt.show()
    return fig

def plot_histogram_from_csv(csv_path, plot_label='circle_dia/um', bin_s=100):
    stati = pd.read_csv(csv_path)
    # 统计均值、标准差、D50、D90、D99保存于stati_data中
    stati_data = round(pd.concat([stati.mean(), stati.std(), stati.median(), stati.quantile(0.9), stati.quantile(0.99)], axis=1).T, 1)
    stati_data.index = ['average', 'standardization', 'D50', 'D90', 'D99']
    
    plt.figure(figsize=(15, 3))
    hist, bins = np.histogram(stati[plot_label], bins=bin_s)
    plt.bar(bins[:-1], hist, width=np.diff(bins), ec='k')

    plt.title('mean:{}  std:{}  max:{}  D50:{}  D90:{}  D99:{}  unit:um'.format(stati_data[plot_label][0],stati_data[plot_label][1],stati[plot_label].max(),stati_data[plot_label][2],stati_data[plot_label][3],stati_data[plot_label][4])) 
    plt.grid(axis='y', alpha=0.3)
    plt.xlabel('particle size(um)')
    plt.ylabel('frequency')
    plt.tight_layout()
    plt.show()

# 计算dinger-funk方程的CPFT
def dinger_funk_CPFT(D, Ds, Dl, n=0.333):
    if D < Ds:
        return 0.
    elif (D >= Ds) and (D <= Dl):
        numerator = D**n - Ds**n
        denominator = Dl**n - Ds**n
        CPFT = round(100*numerator/denominator, 2)
        print('Diameter: {} CPFT(%): {}'.format(D,CPFT))
        return CPFT
    elif D > Dl:
        return 100.

# 根据包含粒径分布的dataframe, 生成并返回新的CPFT数据
def distr_to_CPFT(pris_df, col_name=None):
    # 先对传入的数据按列进行归一化，再乘以100
    column_sums = pris_df.sum()
    print(column_sums, type(column_sums))
    if (column_sums == 100.).all():
        normalized_pris_df = pris_df
    elif not (column_sums == 100.).all():
        normalized_pris_df = pris_df.div(0.01 * column_sums, axis=1)
    CPFT_df = pd.DataFrame()
    if isinstance(col_name, str):
        # 如果指定的列名是一个字符串，则只对dataframe指定列进行累计求和
        CPFT_df[str(col_name)+'_CPFT'] = normalized_pris_df[col_name].cumsum()
    elif isinstance(col_name, list):
        # 如果指定的列名是一个list，则依次累积求和
        for col in col_name:
            CPFT_df[str(col)+'_CPFT'] = normalized_pris_df[col].cumsum()
    elif col_name is None:
        # 未指定列名，默认对所有列进行累计求和
        CPFT_df = normalized_pris_df.cumsum()
        CPFT_df = CPFT_df.add_suffix('_CPFT')   # 将列名修改为带有后缀的形式
        # for col in normalized_pris_df.columns:
        #     CPFT_df[str(col)+'_CPFT'] = normalized_pris_df[col].cumsum()
    return CPFT_df

# 统计落在 'range_list' 列中各个数字之间的颗粒数量，##默认与Sieves大小相等的颗粒无法过筛##
def count_particle_from_range(distr_df, range_list, col_name=None):
    count_dict = {}
    if isinstance(col_name, str):
        # 如果col_name是个string，只对dataframe指定列进行计数
        count = distr_df[distr_df[col_name] < range_list[0]][col_name].count()
        count_dict[f'{range_list[0]}'] = count
        for i in range(len(range_list)-1):
            count = distr_df[(distr_df[col_name] >= range_list[i]) & (distr_df[col_name] < range_list[i+1])][col_name].count()
            count_dict[f'{range_list[i+1]}'] = count
        reCount_df = pd.DataFrame(list(count_dict.items()), columns=['Sieves', col_name+'Count']).astype(float)
    elif isinstance(col_name, list):
        # 如果指定的列名是一个list，则依次计数
        reCount_df = pd.DataFrame({'Sieves':range_list})
        for col in col_name:
            temp = {}
            count = distr_df[distr_df[col] < range_list[0]][col].count()
            temp[f'{range_list[0]}'] = count
            for i in range(len(range_list)-1):
                count = distr_df[(distr_df[col] >= range_list[i]) & (distr_df[col] < range_list[i+1])][col].count()
                temp[f'{range_list[i+1]}'] = count
            temp = pd.DataFrame(list(temp.items()), columns=['Sieves', col+'Count'])
            temp = temp.astype(float)
            reCount_df = pd.merge(temp, reCount_df, on='Sieves', how='outer')
    elif col_name is None:
        # 如果未指定，则默认对所有列计数
        reCount_df = pd.DataFrame({'Sieves':range_list})
        for col in distr_df.columns:
            temp = {}
            count = distr_df[distr_df[col] < range_list[0]][col].count()
            temp[f'{range_list[0]}'] = count
            for i in range(len(range_list)-1):
                count = distr_df[(distr_df[col] >= range_list[i]) & (distr_df[col] < range_list[i+1])][col].count()
                temp[f'{range_list[i+1]}'] = count
            temp = pd.DataFrame(list(temp.items()), columns=['Sieves', col+'Count'])
            # print(temp)
            temp = temp.astype(float)
            reCount_df = pd.merge(temp, reCount_df, on='Sieves', how='outer')
    return reCount_df

# 计算各种配比的双组份混合粉体与根据model计算的最密堆积间的相关系数
def corrcoef_to_model(goal_df, model_df):
    # 计算所有列与model的相关系数
    corr_series = goal_df.corrwith(model_df.iloc[:,1])
    # 将结果转换为DataFrame
    corr_df = pd.DataFrame(corr_series, columns=['R^2'])

    return corr_df.T
