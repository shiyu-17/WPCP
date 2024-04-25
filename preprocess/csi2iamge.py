from CSIKit.visualization.metric import *
from CSIKit.visualization.plot_scenario import *
import matplotlib.pyplot as plt
import numpy as np
import os

input_dir = "E:\dataset\pic_960\csi"  
output_dir = "E:\dataset\pic_960\csi_pic" 

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

my_reader = IWLBeamformReader()

for filename in os.listdir(input_dir):
    if filename.endswith(".dat"):
        csi_data = my_reader.read_file(os.path.join(input_dir, filename))

        csi_list = []
        for i in range(500, 750):
            csi_list.append(csi_data.frames[i].csi_matrix[:, :, 0]) #1000*30*1
        csi_data = np.array(csi_list)
        csi_matrix_am = np.abs(csi_data)

        plt.figure()
        plt.pcolormesh(np.transpose(csi_matrix_am[:, :, 0]), cmap='viridis_r')
        plt.savefig(os.path.join(output_dir, os.path.splitext(filename)[0] + ".png"))
        plt.close() 
        print(filename)

print("图像生成并成功保存。")
