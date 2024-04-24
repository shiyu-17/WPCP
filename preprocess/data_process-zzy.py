from CSIKit.visualization.metric import *
from CSIKit.visualization.plot_scenario import *
import matplotlib.pyplot as plt
import numpy as np
import os

output_folder = ".\csi-pic"
folder_path = r"E:\Mycode\WPCP\raw"

for file_name in os.listdir(folder_path):
    if file_name.endswith(".dat"):
        print(file_name)
        my_reader = IWLBeamformReader()
        csi_data = my_reader.read_file(file_name)
        csi_list = []
        for i in range(500, 750):
            csi_list.append(csi_data.frames[i].csi_matrix[::, ::, 0]) #1000*30*3*1
        csi_data = np.array(csi_list)
        csi_matrix_am = np.abs(csi_data)

        print(csi_matrix_am.shape)


        plt.pcolormesh(np.transpose(csi_matrix_am[:: ,::, 0]), cmap='viridis_r')

        output_file_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".png")
        plt.savefig(output_file_path)
        plt.show()

