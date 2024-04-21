from CSIKit.visualization.metric import *
from CSIKit.visualization.plot_scenario import *
import matplotlib.pyplot as plt

import numpy as np
my_reader = IWLBeamformReader()
csi_data = my_reader.read_file("hb-2-2-4-1-1-c01.dat")
csi_list = []
# for i in range(len(csi_data.frames)):
for i in range(1000):
    # print(csi_data.frames[i].csi_matrix[::, ::, ::].shape)
    csi_list.append(csi_data.frames[i].csi_matrix[::, ::, 0]) #1000*30*3*1
csi_data = np.array(csi_list)
# csi_data = csi_data[::,::,::,0]/csi_data[::,::,::,1] # 天线商校准相位
csi_matrix_am = np.abs(csi_data)
# csi_matrix_ph = np.angle(csi_data)

print(csi_matrix_am.shape)

#plt.plot(csi matrix_am[::.::.0l,linewidth=0.8)
plt.pcolormesh( np.transpose(csi_matrix_am[:: ,::, 0]), cmap='viridis_r')
#print(np.transpose(csi_matrix_am[:: ,::, 0]).shape)
# plt.axis('off')
plt.savefig( './')
# plt.plot(csi_matrix_am[::,2,0], linewidth=0.8)
#plt.show()

