import numpy as np



file_train = "/media/davi/dados/Projetos/CamNuvem/dataset/ucf_crimes/i3d/i3d_train/Abuse001_x264_i3d.npy"
file_test = "/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/CamNuvem_dataset_normalizado/i3d-10-cropped/test/anomaly/10.npy"



data = np.load(file_test)
print(data.shape)