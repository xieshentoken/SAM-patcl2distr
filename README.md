# SAM-patcl2distr
## 简介
一个基于segment-anything的粒径统计工具。
## 基本功能
1. 手动选择图像的标尺长度，读取标尺像素信息和物理尺寸信息后保存在config文件夹中；
2. 调用segment-anything进行图像分割（此处也可调用自己训练的模型进行分割），结果保存在export data文件夹中；
3. 通过app文件夹中的外部软件或自带的小程序（根据Andreson-Anderson Model或Dinger-Funk Model）对粒径分布进行优化。
