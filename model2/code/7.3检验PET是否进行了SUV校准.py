import SimpleITK as sitk
import numpy as np

# 替换为你的PET文件路径
#pet_path = r"D:\gulianyu\LungAd_Radiomics\pet_suv\1016\1016_PET_SUV.nii.gz"
pet_path = r"D:\gulianyu\LungAd_Radiomics\registered_nifti\1016\1016_PET_registered.nii.gz"

pet_image = sitk.ReadImage(pet_path)
pet_array = sitk.GetArrayFromImage(pet_image)
print(f"PET像素值范围：{np.min(pet_array)} ~ {np.max(pet_array)}")
print(f"PET像素值均值：{np.mean(pet_array[pet_array>0])}")  # 仅看非背景区域均值