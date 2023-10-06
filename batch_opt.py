import patcl2distr_library as p2d
import config as scf

import PySimpleGUI as sg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class CSVSelector:
    def __init__(self):
        # 初始化自变量
        self.file_path = None
        self.selected_columns = None
        self.selected_data = None
        self.particleSize_max = None
        self.particleSize_min = None
        self.distri_modulus = 0.375
        self.dual_mixture = pd.DataFrame()
        self.min_mix_ratio1 = 0
        self.max_mix_ratio1 = 1.0
        self.weight_number1 = 50
        self.min_mix_ratio2 = 0
        self.max_mix_ratio2 = 1.0
        self.weight_number2 = 25
        self.min_mix_ratio3 = 0
        self.max_mix_ratio3 = 1.0
        self.weight_number3 = 25
        
    def make_window(self):
        self.layout = [
            [sg.Button('Select CSV File', key='-OPEN-'),
                sg.Button('ReSelect Columns', key='-ReOPEN-')],
            [sg.Text('', key='-TEXT-', font=('Helvetica', 14))],
            [sg.Button('Dinger-Funk Model', key='-DFM_Fit-'), sg.Button('New Project', key='-NEWP-')]
        ]
        return sg.Window('CSV Selector', self.layout)
        
    def open_csv(self):
        if self.file_path:
            df = pd.read_csv(self.file_path)
            csv_name = self.file_path.split('/')[-1].split('.csv')[0]
            self.window['-TEXT-'].update(f'Database of particle distribution: {csv_name}')
            col_names = list(df.columns)
            popup_layout = [
                [sg.Listbox(col_names, size=(20, len(col_names)), key='-LISTBOX-', select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE)],
                [sg.Button('OK', key='-OK-')]
            ]
            popup_window = sg.Window('Select Columns', popup_layout)
            self.window.hide()  # 隐藏主窗口
            while True:
                event, values = popup_window.read()
                if event == sg.WIN_CLOSED:
                    break
                if event == '-OK-':
                    self.selected_columns = values['-LISTBOX-']
                    self.selected_data = pd.concat([df.iloc[:,0],df[self.selected_columns]], axis=1)  # 选择不同粉体，csv第一行是粉体批号，第二行是对应振实密度
                    rows_to_delete = list(self.selected_data.index[:2])
                    for index, row in self.selected_data.iloc[2:,1:].iterrows():
                        if (row == 0).all():
                            rows_to_delete.append(index)
                    self.particleSize_max = self.selected_data.drop(rows_to_delete).iloc[-1,0]
                    self.particleSize_min = self.selected_data.drop(rows_to_delete).iloc[0,0]
                    print(self.selected_columns, type(self.selected_data.iloc[1,2]))
                    popup_window.close()  # 关闭Listbox窗口
                    self.window.un_hide()  # 显示主窗口
                    break
    
    def config_dinger_funk(self):
        layout = [
            [sg.Text('目前只能选择2~3种粉体进行优化，其他功能待开发...', key='-TIPS-', font=('Helvetica', 14))],
            [sg.Text('max size'), sg.InputText(self.particleSize_max, key='-MAX_SIZE-')],
            [sg.Text('min size'), sg.InputText(self.particleSize_min, key='-MIN_SIZE-')],
            [sg.Text('modulus'), sg.InputText(self.distri_modulus, key='-MODULUS-')],
            [sg.Button('OK', key='-OK-'), sg.Button('Calculate', key ='-Cal-'), sg.Button('Mix Ratio', key ='-SetW-')]
        ]
        window = sg.Window('Dinger funk model parameter', layout)
        while True:
            event1, values1 = window.read()
            if event1 == sg.WIN_CLOSED:
                break
            if event1 == '-OK-':
                self.particleSize_max = float(values1['-MAX_SIZE-'])
                self.particleSize_min = float(values1['-MIN_SIZE-'])
                self.distri_modulus = float(values1['-MODULUS-'])
            if event1 == '-Cal-':
                self.particleSize_max = float(values1['-MAX_SIZE-'])
                self.particleSize_min = float(values1['-MIN_SIZE-'])
                self.distri_modulus = float(values1['-MODULUS-'])
                self.optimization_via_MAA()
            if event1 == '-SetW-':
                self.set_calcu_range()

    def set_calcu_range(self):
        if self.selected_data.shape[1] == 3:
            layout = [
                [sg.Text('选择两种组分混合的比例范围（0-1.0）', key='-TIP2-', font=('Helvetica', 14))],
                [sg.Text('Min ratio'), sg.InputText(self.min_mix_ratio1, key='-MIN_RATIO-', size=(7, 1))],
                [sg.Text('Max ratio'), sg.InputText(self.max_mix_ratio1, key='-MAX_RATIO-', size=(7, 1))],
                [sg.Text('Segments'), sg.InputText(self.weight_number1, key='-WEIGHT_NUM-', size=(7, 1))],
                [sg.Button('OK', key='-OK2-')]
            ]
            window = sg.Window('Set calculate range', layout)
            event2, values2 = window.read()
            if event2 == '-OK2-':
                self.min_mix_ratio1 = float(values2['-MIN_RATIO-'])
                self.max_mix_ratio1 = float(values2['-MAX_RATIO-'])
                self.weight_number = int(values2['-WEIGHT_NUM-'])
        elif self.selected_data.shape[1] == 4:
            self.weight_number1 = 25
            self.weight_number2 = 25
            layout = [
                [sg.Text('选择混合的比例范围（0-1.0）,三组分之和为1.0', key='-TIP2-', font=('Helvetica', 14))],
                [sg.Text(self.selected_columns[0]+'Min ratio'), sg.InputText(self.min_mix_ratio1, key='-MIN_RATIO_1-', size=(7, 1)),
                 sg.Text(self.selected_columns[1]+'Min ratio'), sg.InputText(self.min_mix_ratio1, key='-MIN_RATIO_2-', size=(7, 1))],
                [sg.Text(self.selected_columns[0]+'Max ratio'), sg.InputText(self.max_mix_ratio1, key='-MAX_RATIO_1-', size=(7, 1)),
                 sg.Text(self.selected_columns[1]+'Max ratio'), sg.InputText(self.max_mix_ratio1, key='-MAX_RATIO_2-', size=(7, 1))],
                [sg.Text(self.selected_columns[0]+'Segments'), sg.InputText(self.weight_number1, key='-WEIGHT_NUM_1-', size=(7, 1)),
                 sg.Text(self.selected_columns[1]+'Segments'), sg.InputText(self.weight_number2, key='-WEIGHT_NUM_2-', size=(7, 1))],
                [sg.Button('OK', key='-OK2-')]
            ]
            window = sg.Window('Set calculate range', layout)
            event2, values2 = window.read()
            if event2 == '-OK2-':
                self.min_mix_ratio1 = float(values2['-MIN_RATIO_1-'])
                self.max_mix_ratio1 = float(values2['-MAX_RATIO_1-'])
                self.min_mix_ratio2 = float(values2['-MIN_RATIO_2-'])
                self.max_mix_ratio2 = float(values2['-MAX_RATIO_2-'])
                self.weight_number1 = int(values2['-WEIGHT_NUM_1-'])
                self.weight_number2 = int(values2['-WEIGHT_NUM_2-'])
            if (self.min_mix_ratio1+self.min_mix_ratio2) > 1.:
                sg.popup('组分总含量合计超过1，请重新输入', title='Error')
        elif self.selected_data.shape[1] == 5:
            self.weight_number1 = 25
            self.weight_number2 = 25
            self.weight_number3 = 25
            layout = [
                [sg.Text('选择混合的比例范围（0-1.0）,四组分之和为1.0', key='-TIP2-', font=('Helvetica', 14))],
                [sg.Text(self.selected_columns[0]+'Min ratio'), sg.InputText(self.min_mix_ratio1, key='-MIN_RATIO_1-', size=(7, 1)),
                 sg.Text(self.selected_columns[1]+'Min ratio'), sg.InputText(self.min_mix_ratio1, key='-MIN_RATIO_2-', size=(7, 1)), 
                 sg.Text(self.selected_columns[2]+'Min ratio'), sg.InputText(self.min_mix_ratio1, key='-MIN_RATIO_3-', size=(7, 1))],
                [sg.Text(self.selected_columns[0]+'Max ratio'), sg.InputText(self.max_mix_ratio1, key='-MAX_RATIO_1-', size=(7, 1)),
                 sg.Text(self.selected_columns[1]+'Max ratio'), sg.InputText(self.max_mix_ratio1, key='-MAX_RATIO_2-', size=(7, 1)),
                 sg.Text(self.selected_columns[2]+'Max ratio'), sg.InputText(self.max_mix_ratio1, key='-MAX_RATIO_3-', size=(7, 1))],
                [sg.Text(self.selected_columns[0]+'Segments'), sg.InputText(self.weight_number1, key='-WEIGHT_NUM_1-', size=(7, 1)),
                 sg.Text(self.selected_columns[1]+'Segments'), sg.InputText(self.weight_number2, key='-WEIGHT_NUM_2-', size=(7, 1)),
                 sg.Text(self.selected_columns[2]+'Segments'), sg.InputText(self.weight_number2, key='-WEIGHT_NUM_3-', size=(7, 1))],
                [sg.Button('OK', key='-OK3-')]
            ]
            window = sg.Window('Set calculate range', layout)
            event3, values3 = window.read()
            if event3 == '-OK3-':
                self.min_mix_ratio1 = float(values3['-MIN_RATIO_1-'])
                self.max_mix_ratio1 = float(values3['-MAX_RATIO_1-'])
                self.min_mix_ratio2 = float(values3['-MIN_RATIO_2-'])
                self.max_mix_ratio2 = float(values3['-MAX_RATIO_2-'])
                self.min_mix_ratio3 = float(values3['-MIN_RATIO_3-'])
                self.max_mix_ratio3 = float(values3['-MAX_RATIO_3-'])
                self.weight_number1 = int(values3['-WEIGHT_NUM_1-'])
                self.weight_number2 = int(values3['-WEIGHT_NUM_2-'])
                self.weight_number3 = int(values3['-WEIGHT_NUM_3-'])
            if (self.min_mix_ratio1+self.min_mix_ratio2+self.min_mix_ratio3) > 1.:
                sg.popup('组分总含量合计超过1，请重新输入', title='Error')
        else:
            sg.popup('只能选择2~4种粉体，多组分拟合功能正在开发中...', title='Error')
        window.close()

    def optimization_via_MAA(self):
        density_dict = self.selected_data.iloc[0,1:].to_dict()        # 获取粉体的振实密度值
        # 根据设定的分布模量计算Dinger-Funk最密堆积的CPFT曲线
        self.model_CPFT = self.selected_data.iloc[1:,0].apply(lambda x: p2d.dinger_funk_CPFT(float(x), self.particleSize_min, self.particleSize_max, self.distri_modulus))
        self.model_CPFT = pd.concat([self.selected_data.iloc[1:,0], self.model_CPFT], axis=1)
        self.model_CPFT.columns = ['Sieves_um', 'CPFT(n='+str(self.distri_modulus)+' of Dinger_Funk)']
        if self.selected_data.shape[1] == 3:
            self.dual_mixture = pd.DataFrame()
            weight = np.round(np.linspace(self.min_mix_ratio1, self.max_mix_ratio1, self.weight_number1), 4)     # 定义粉体的权重参数分布范围
            for w1 in weight:
                self.dual_mixture[w1] = w1 * self.selected_data[self.selected_columns[0]] + (1 - w1) * self.selected_data[self.selected_columns[1]]
            self.dual_mixture = p2d.distr_to_CPFT(self.dual_mixture.iloc[1:,:]).round(4)
            self.dual_mixture = pd.concat([self.selected_data.iloc[1:,0], self.dual_mixture], axis=1)

            # self.dual_mixture.to_csv('/Users/helong/code learning/packing/weight.csv', index=False)
            corrcoef_df = p2d.corrcoef_to_model(self.dual_mixture.iloc[:,1:], self.model_CPFT)
            opti_ratio = corrcoef_df.idxmax(axis=1)[0]#.split('_CPFT')[0]   # string
            corrcoef_df.columns = [i for i in range(0, self.weight_number1, 1)]
            self.weight_df = pd.DataFrame(weight)
            self.weight_df['R^2'] = corrcoef_df.T
            
            # self.model_CPFT.to_csv('/Users/helong/code learning/packing/dingerfunk_model.csv', index=False)
            # self.weight_df.to_csv('/Users/helong/code learning/packing/corrcoef.csv', index=False)
            # 设置Seaborn样式和调色板
            sns.set(style='white', palette='muted')  # 去掉网格线，使用muted调色板
            # 创建一个1x2的图形布局
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            # 在第一个子图中绘制图表
            sns.lineplot(x=self.weight_df[self.weight_df.columns[0]], y=self.weight_df['R^2'], ax=axes[0], label='Data Points', marker='o')  # 设置点的大小
            axes[0].set_title('R^2 of '+str(self.selected_columns[0])+'+'+str(self.selected_columns[1]))
            axes[0].set_xlabel(str(self.selected_columns[0])+" Mixture Ratio")  # 设置X轴标签
            axes[0].set_ylabel('corrcoef R^2')  # 设置Y轴标签
            # 在第二个子图中绘制图表
            sns.lineplot(x=self.dual_mixture.iloc[:,0], y=self.dual_mixture[opti_ratio], ax=axes[1], palette='husl', label=opti_ratio)  # 使用husl调色板
            axes[1].set_title('CPFT of optimization mixture ratio')
            axes[1].set_xlabel('Sieves Size(um)')  # 设置X轴标签
            axes[1].set_ylabel('CPFT (%)')  # 设置Y轴标签
            axes[1].set_xscale('log')  # 设置对数刻度
            sns.lineplot(x=self.dual_mixture.iloc[:,0], y=self.model_CPFT.iloc[:,1], 
                            ax=axes[1], palette='husl', label='n: '+str(self.distri_modulus)+'(Dinger-Funk Model)')  # 使用husl调色板

            # 设置图例位置
            axes[0].legend(loc='best')
            axes[1].legend(loc='best')

            # 调整子图之间的间距
            plt.tight_layout()

            # 显示图形
            plt.show()
        elif self.selected_data.shape[1] == 4:
            self.trip_mixture = pd.DataFrame()
            weight1 = np.round(np.linspace(self.min_mix_ratio1, self.max_mix_ratio1, self.weight_number1), 4)     # 定义粉体的权重参数分布范围
            weight2 = np.round(np.linspace(self.min_mix_ratio2, self.max_mix_ratio2, self.weight_number2), 4)
            weight_group = [(w1, w2) for w1 in weight1 for w2 in weight2 if w1 + w2 <= 1.0]
            # 计算所有权重组合的结果并添加到DataFrame中
            self.trip_mixture = pd.concat([w[0] * self.selected_data[self.selected_columns[0]] + 
                                        w[1] * self.selected_data[self.selected_columns[1]] + 
                                        (1 - w[0] - w[1]) * self.selected_data[self.selected_columns[2]]
                                        for w in weight_group], axis=1)
            # 为列添加相应的列名
            self.trip_mixture.columns = [f'{w[0]}_{w[1]}' for w in weight_group]

            self.trip_mixture = p2d.distr_to_CPFT(self.trip_mixture.iloc[1:,:]).round(4)
            self.trip_mixture = pd.concat([self.selected_data.iloc[1:,0], self.trip_mixture], axis=1)
            corrcoef_df = p2d.corrcoef_to_model(self.trip_mixture.iloc[:,1:], self.model_CPFT)
            opti_ratio = corrcoef_df.idxmax(axis=1)[0]#.split('_CPFT')[0]   # string
            # print(self.trip_mixture.columns, opti_ratio)
            corrcoef_df.columns = [i for i in range(0, len(weight_group), 1)]
            self.weight_df = pd.DataFrame(weight_group)
            self.weight_df.columns = [self.selected_columns[0],self.selected_columns[1]]
            self.weight_df['R^2'] = corrcoef_df.T
            # self.weight_df.to_csv('/Users/helong/code learning/packing/corrcoef.csv', index=True)
            X = self.weight_df[self.selected_columns[0]]
            Y = self.weight_df[self.selected_columns[1]]
            Z = self.weight_df['R^2']
            cmap = plt.get_cmap('coolwarm')
            # 创建散点图
            fig = plt.figure(figsize=(12, 8))
            ax1 = fig.add_subplot(221)
            scatter = ax1.scatter(X, Y, c=Z, cmap=cmap, alpha=0.7)
            # 找到最大值的坐标
            max_index = Z.idxmax()
            max_x = X[max_index]
            max_y = Y[max_index]
            max_z = Z[max_index]
            print(max_x,max_y,max_z)
            cbar = fig.colorbar(scatter, ax=ax1)
            cbar.set_label('R^2')
            ax1.set_xlabel(self.selected_columns[0])
            ax1.set_ylabel(self.selected_columns[1])
            ax1.set_title(f'Max R^2 Value: {max_z:.2f} at \n({self.selected_columns[0]}: {max_x*100}%, \n{self.selected_columns[1]}: {max_y*100}%, \n{self.selected_columns[2]}: {(1-max_x-max_y)*100}%)')
            # 绘制子图
            x_line = self.trip_mixture.iloc[:,0]
            y1_line = self.model_CPFT.iloc[:,1]
            y2_line = self.trip_mixture[opti_ratio]
            ax2 = fig.add_subplot(222)
            ax2.plot(x_line, y1_line, color='blue', label=f'n: {self.distri_modulus} (Dinger-Funk Model)')
            ax2.plot(x_line, y2_line, color='red', label=f'A: {max_x*100}%-B: {max_y*100}%-C: {(1-max_x-max_y)*100}%')
            ax2.set_title('Variation to ideal Dinger-Funk model')
            ax2.set_xlabel('Sieves Size(um)')
            ax2.set_ylabel('CPFT (%)')
            
            select_batch_df = self.selected_data.iloc[1:,:].astype(float)
            # select_batch_df.to_csv('/Users/helong/code learning/packing/select_data.csv',index=True)
            ax3 = fig.add_subplot(234)
            ax3.plot(x_line, p2d.distr_to_CPFT(select_batch_df, self.selected_columns[0]).iloc[:,0], color='blue', label=f'{self.selected_columns[0]} CPFT')
            ax3.set_xlabel('Sieves Size(um)')
            ax3.set_ylabel('CPFT (%)', color='blue')
            ax3.tick_params(axis='y', labelcolor='blue')
            ax3_t = ax3.twinx()
            ax3_t.bar(x_line, select_batch_df[self.selected_columns[0]], color='green', alpha=0.5)
            # ax3_t.set_ylabel('Counts', color='green')
            ax3_t.tick_params(axis='y', labelcolor='green')
            ax3.set_title(f'{self.selected_columns[0]}')

            ax4 = fig.add_subplot(235)
            ax4.plot(x_line, p2d.distr_to_CPFT(select_batch_df, self.selected_columns[1]).iloc[:,0], color='blue', label=f'{self.selected_columns[1]} CPFT')
            ax4.set_xlabel('Sieves Size(um)')
            # ax4.set_ylabel('CPFT (%)', color='blue')
            ax4.tick_params(axis='y', labelcolor='blue')
            ax4_t = ax4.twinx()
            ax4_t.bar(x_line, select_batch_df[self.selected_columns[1]], color='green', alpha=0.5)
            # ax4_t.set_ylabel('Counts', color='green')
            ax4_t.tick_params(axis='y', labelcolor='green')
            ax4.set_title(f'{self.selected_columns[1]}')

            ax5 = fig.add_subplot(236)
            ax5.plot(x_line, p2d.distr_to_CPFT(select_batch_df, self.selected_columns[2]).iloc[:,0], color='blue', label=f'{self.selected_columns[2]} CPFT')
            ax5.set_xlabel('Sieves Size(um)')
            # ax5.set_ylabel('CPFT (%)', color='blue')
            ax5.tick_params(axis='y', labelcolor='blue')
            ax5_t = ax5.twinx()
            ax5_t.bar(x_line, select_batch_df[self.selected_columns[2]], color='green', alpha=0.5)
            ax5_t.set_ylabel('Counts', color='green')
            ax5_t.tick_params(axis='y', labelcolor='green')
            ax5.set_title(f'{self.selected_columns[2]}')
            
            ax1.legend(loc='best')
            ax2.legend(loc='best')
            plt.tight_layout()
            plt.show()
        elif self.selected_data.shape[1] == 5:
            self.quadr_mixture = pd.DataFrame()
            weight1 = np.round(np.linspace(self.min_mix_ratio1, self.max_mix_ratio1, self.weight_number1), 4)     # 定义粉体的权重参数分布范围
            weight2 = np.round(np.linspace(self.min_mix_ratio2, self.max_mix_ratio2, self.weight_number2), 4)
            weight3 = np.round(np.linspace(self.min_mix_ratio3, self.max_mix_ratio3, self.weight_number3), 4)
            weight_group = [(w1, w2, w3) for w1 in weight1 for w2 in weight2 for w3 in weight3 if w1 + w2 + w3 <= 1.0]
            # 计算所有权重组合的结果并添加到DataFrame中
            self.quadr_mixture = pd.concat([w[0] * self.selected_data[self.selected_columns[0]] + 
                                        w[1] * self.selected_data[self.selected_columns[1]] + 
                                        w[2] * self.selected_data[self.selected_columns[2]] + 
                                        (1 - w[0] - w[1] - w[2]) * self.selected_data[self.selected_columns[3]]
                                        for w in weight_group], axis=1)
            # 为列添加相应的列名
            self.quadr_mixture.columns = [f'{w[0]}_{w[1]}_{w[2]}' for w in weight_group]

            self.quadr_mixture = p2d.distr_to_CPFT(self.quadr_mixture.iloc[1:,:]).round(4)
            self.quadr_mixture = pd.concat([self.selected_data.iloc[1:,0], self.quadr_mixture], axis=1)
            corrcoef_df = p2d.corrcoef_to_model(self.quadr_mixture.iloc[:,1:], self.model_CPFT)
            opti_ratio = corrcoef_df.idxmax(axis=1)[0]
            # print(self.quadr_mixture.columns, opti_ratio)
            corrcoef_df.columns = [i for i in range(0, len(weight_group), 1)]
            self.weight_df = pd.DataFrame(weight_group)
            self.weight_df.columns = [self.selected_columns[0], self.selected_columns[1], self.selected_columns[2]]
            self.weight_df['R^2'] = corrcoef_df.T
            # self.weight_df.to_csv('/Users/helong/code learning/packing/corrcoef4.csv', index=True)
            X = self.weight_df[self.selected_columns[0]]
            Y = self.weight_df[self.selected_columns[1]]
            Z = self.weight_df[self.selected_columns[2]]
            colors = self.weight_df['R^2']
            # 创建三维散点图
            fig = plt.figure(figsize=(15, 8))
            cmap = plt.get_cmap('coolwarm')#'viridis'
            ax1 = fig.add_subplot(221, projection='3d')
            scatter = ax1.scatter(X, Y, Z, c=colors, cmap=cmap)  # 使用scatter绘制散点图
            cbar = fig.colorbar(scatter, ax=ax1)
            cbar.set_label('R^2')
            # 找到最大值的坐标
            max_index = np.argmax(colors)
            max_x = X[max_index]
            max_y = Y[max_index]
            max_z = Z[max_index]
            max_color = colors[max_index]
            print(max_x,max_y,max_z)
            ax1.set_xlabel(self.selected_columns[0])
            ax1.set_ylabel(self.selected_columns[1])
            ax1.set_zlabel(self.selected_columns[2])
            ax1.set_title(f'Max R^2 Value: {max_color:.2f} at \n({self.selected_columns[0]}: {max_x*100}%, {self.selected_columns[1]}: {max_y*100}%, \n{self.selected_columns[2]}: {max_z*100}%, {self.selected_columns[3]}: {(1-max_x-max_y-max_z)*100}%)')
            # 绘制子图
            x_line = self.quadr_mixture.iloc[:,0]
            y1_line = self.model_CPFT.iloc[:,1]
            y2_line = self.quadr_mixture[opti_ratio]
            ax2 = fig.add_subplot(222)
            ax2.plot(x_line, y1_line, color='blue', label=f'n: {self.distri_modulus} (Dinger-Funk Model)')
            ax2.plot(x_line, y2_line, color='red', label=f'A: {max_x*100}%-B: {max_y*100}%-C: {max_z*100}%-D: {(1-max_x-max_y-max_z)*100}%')
            ax2.set_title('Variation to ideal Dinger-Funk model')
            ax2.set_xlabel('Sieves Size(um)')
            ax2.set_ylabel('CPFT (%)')
            
            select_batch_df = self.selected_data.iloc[1:,:].astype(float)
            # select_batch_df.to_csv('/Users/helong/code learning/packing/select_data.csv',index=True)
            ax3 = fig.add_subplot(245)
            ax3.plot(x_line, p2d.distr_to_CPFT(select_batch_df, self.selected_columns[0]).iloc[:,0], color='blue', label=f'{self.selected_columns[0]} CPFT')
            ax3.set_xlabel('Sieves Size(um)')
            ax3.set_ylabel('CPFT (%)', color='blue')
            ax3.tick_params(axis='y', labelcolor='blue')
            ax3_t = ax3.twinx()
            ax3_t.bar(x_line, select_batch_df[self.selected_columns[0]], color='green', alpha=0.5)
            # ax3_t.set_ylabel('Counts', color='green')
            ax3_t.tick_params(axis='y', labelcolor='green')
            ax3.set_title(f'{self.selected_columns[0]}')

            ax4 = fig.add_subplot(246)
            ax4.plot(x_line, p2d.distr_to_CPFT(select_batch_df, self.selected_columns[1]).iloc[:,0], color='blue', label=f'{self.selected_columns[1]} CPFT')
            ax4.set_xlabel('Sieves Size(um)')
            # ax4.set_ylabel('CPFT (%)', color='blue')
            ax4.tick_params(axis='y', labelcolor='blue')
            ax4_t = ax4.twinx()
            ax4_t.bar(x_line, select_batch_df[self.selected_columns[1]], color='green', alpha=0.5)
            # ax4_t.set_ylabel('Counts', color='green')
            ax4_t.tick_params(axis='y', labelcolor='green')
            ax4.set_title(f'{self.selected_columns[1]}')

            ax5 = fig.add_subplot(247)
            ax5.plot(x_line, p2d.distr_to_CPFT(select_batch_df, self.selected_columns[2]).iloc[:,0], color='blue', label=f'{self.selected_columns[2]} CPFT')
            ax5.set_xlabel('Sieves Size(um)')
            # ax5.set_ylabel('CPFT (%)', color='blue')
            ax5.tick_params(axis='y', labelcolor='blue')
            ax5_t = ax5.twinx()
            ax5_t.bar(x_line, select_batch_df[self.selected_columns[2]], color='green', alpha=0.5)
            # ax5_t.set_ylabel('Counts', color='green')
            ax5_t.tick_params(axis='y', labelcolor='green')
            ax5.set_title(f'{self.selected_columns[2]}')
            
            ax6 = fig.add_subplot(248)
            ax6.plot(x_line, p2d.distr_to_CPFT(select_batch_df, self.selected_columns[3]).iloc[:,0], color='blue', label=f'{self.selected_columns[3]} CPFT')
            ax6.set_xlabel('Sieves Size(um)')
            # ax6.set_ylabel('CPFT (%)', color='blue')
            ax6.tick_params(axis='y', labelcolor='blue')
            ax6_t = ax6.twinx()
            ax6_t.bar(x_line, select_batch_df[self.selected_columns[3]], color='green', alpha=0.5)
            ax6_t.set_ylabel('Counts', color='green')
            ax6_t.tick_params(axis='y', labelcolor='green')
            ax6.set_title(f'{self.selected_columns[3]}')
            
            ax1.legend(loc='best')
            ax2.legend(loc='best')
            plt.tight_layout()
            plt.show()
        else:
            sg.popup('只能选择2~4种粉体,多组分拟合功能正在开发中...', title='Error')

    def run(self):
        self.window = self.make_window()
        while True:
            event, values = self.window.read()
            print(event)
            if event == sg.WIN_CLOSED:
                break
            if event == '-NEWP-':
                self.window.close()
                self.window = self.make_window()
                event, values = self.window.read()
            if event == '-OPEN-':
                default_path = scf.defualt_database_path
                self.file_path = sg.popup_get_file('Select CSV File', initial_folder=default_path, file_types=(("CSV Files", "*.csv"),))
                self.open_csv()
            if event == '-ReOPEN-':
                self.open_csv()
            if event == '-DFM_Fit-':
                if self.particleSize_max is not None and self.particleSize_min is not None:
                    self.config_dinger_funk()
                else:
                    sg.popup('Particle size not calculated yet', title='Error')
        
        self.window.close()

# 实例化类并运行程序
# csv_selector = CSVSelector()
# csv_selector.run()
