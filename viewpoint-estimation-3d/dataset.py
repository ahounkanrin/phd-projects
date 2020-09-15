import nibabel as nib
import numpy as np
ctscan = nib.load("/scratch/hnkmah001/Datasets/ctfullbody/SMIR.Body.021Y.M.CT.57761.nii")
data = ctscan.get_fdata()
data = np.squeeze(data)
data_up = data[:,:,:512]
print("whole data shape:", data.shape)
print("subset of data:", data_up.shape)