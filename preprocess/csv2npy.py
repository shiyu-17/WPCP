import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_pseudocolor_image(df, cmap='viridis_r'):
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(df, cmap=cmap)
    plt.axis('off')

    # 获取当前图形的图像数据
    plt.tight_layout(pad=0)
    plt.draw()
    img = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
    plt.close()  # 关闭图形，避免内存泄漏

    return img


def convert_to_grayscale_and_flatten(img):
    # 转换为灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray_img


def generate_pseudocolor_maps_to_npy(src_dir, dest_dir):
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

                # 分别为每个列集生成伪彩色图像，转换为灰度图像并添加到npy_img中
                npy_img = []
                img_2 = generate_pseudocolor_image(cols_3n_minus_2.transpose())
                gray_flattened_img_2 = convert_to_grayscale_and_flatten(img_2)
                npy_img.append(gray_flattened_img_2)

                img_1 = generate_pseudocolor_image(cols_3n_minus_1.transpose())
                gray_flattened_img_1 = convert_to_grayscale_and_flatten(img_1)
                npy_img.append(gray_flattened_img_1)

                img_0 = generate_pseudocolor_image(cols_3n.transpose())
                gray_flattened_img_0 = convert_to_grayscale_and_flatten(img_0)
                npy_img.append(gray_flattened_img_0)

                npy_img_array = np.array(npy_img)

                # 构建目标文件路径，保持与源文件同名
                dest_file = os.path.join(dest_dir, os.path.splitext(file)[0] + '.npy')
                print(dest_file)
                np.save(dest_file, npy_img_array)


if __name__ == "__main__":
    import cv2

    # 定义源目录路径和目标目录路径
    source_directory = r"E:\A504-processed\train-csv"
    destination_directory = r"E:\A504-processed\train-npy"

    # 调用函数生成伪彩色图并存储为npy文件
    generate_pseudocolor_maps_to_npy(source_directory, destination_directory)

    # 加载并显示生成的npy文件
    # 例如，加载并显示第一个生成的npy文件
    # sample_npy_file = os.path.join(destination_directory, 'hb-2-2-4-1-1-c02-1.npy')
    # images = np.load(sample_npy_file)
    # print(images)
    # print(images.shape)
    # for i in range(images.shape[0]):
    #     print(images[i].shape)
    #     print(images[i])
    #     plt.axis('off')
    #     plt.imshow(images[i])  # 假设每个灰度图像展平后为640000（800x800）
    #     plt.show()
