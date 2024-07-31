import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 定义输入和输出目录路径
input_dir = r'E:\A504-processed\train-csv'
output_dir = r'E:\A504-processed\train-csipng'

# 创建输出目录来保存伪彩色图
os.makedirs(output_dir, exist_ok=True)

# 遍历输入目录中的CSV文件
for subdir, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.csv'):
            print(file)
            file_path = os.path.join(subdir, file)
            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 提取3n-2列、3n-1列和3n列，提取同一天线的子载波
            cols_3n_minus_2 = df.iloc[:, [i for i in range(df.shape[1]) if (i + 1) % 3 == 1]]
            cols_3n_minus_1 = df.iloc[:, [i for i in range(df.shape[1]) if (i + 1) % 3 == 2]]
            cols_3n = df.iloc[:, [i for i in range(df.shape[1]) if (i + 1) % 3 == 0]]

            # 将这些列拼接在一起
            concatenated_df = pd.concat([cols_3n_minus_2, cols_3n_minus_1, cols_3n], axis=1)

            # 交换横轴和纵轴
            transposed_df = concatenated_df.transpose()

            # 创建伪彩色图
            plt.figure(figsize=(10, 8))
            # 若只保存一个天线的30个子载波，将transposed_df改成22-24行中的一个
            # plt.pcolormesh(cols_3n_minus_2, cmap='viridis_r')
            plt.pcolormesh(transposed_df, cmap='viridis_r')
            # plt.colorbar()  # 添加颜色条
            plt.axis('off')

            # 保存伪彩色图，使用与CSV文件相同的名称
            pseudocolor_filename = os.path.splitext(file)[0] + '.png'
            pseudocolor_path = os.path.join(output_dir, pseudocolor_filename)
            plt.savefig(pseudocolor_path)
            plt.close()

print("CSI2Image Done!")
