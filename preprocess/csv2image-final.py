import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_pseudocolor_map(df, dest_dir, file_name, cmap='viridis_r'):
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(df, cmap=cmap)
    
    # plt.colorbar()  # 可选，为伪彩色图添加颜色条
    plt.savefig(os.path.join(dest_dir, file_name))
    plt.close()  # 关闭图形，避免内存泄漏

def generate_pseudocolor_maps(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    for subdir, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.csv'):
                print(f"正在处理文件: {file}")
                file_path = os.path.join(subdir, file)
                df = pd.read_csv(file_path)

                cols_3n_minus_2 = df.iloc[:, [i for i in range(df.shape[1]) if (i + 1) % 3 == 1]]
                cols_3n_minus_1 = df.iloc[:, [i for i in range(df.shape[1]) if (i + 1) % 3 == 2]]
                cols_3n = df.iloc[:, [i for i in range(df.shape[1]) if (i + 1) % 3 == 0]]

                # 分别为每个列集生成伪彩色图
                generate_pseudocolor_map(cols_3n_minus_2.transpose(), dest_dir, f"{file}_2.png")
                generate_pseudocolor_map(cols_3n_minus_1.transpose(), dest_dir, f"{file}_1.png")
                generate_pseudocolor_map(cols_3n.transpose(), dest_dir, f"{file}_0.png")



if __name__ == "__main__":
    # 定义源目录路径和目标目录路径
    source_directory = "/home/featurize/lsy/data/train-csv"
    destination_directory = "/home/featurize/lsy/data/train-csv-img-3"

    # 调用函数生成伪彩色图
    generate_pseudocolor_maps(source_directory, destination_directory)
