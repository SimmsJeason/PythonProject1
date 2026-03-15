# -*- coding: utf-8 -*-
import os
import SimpleITK as sitk
import numpy as np
import radiomics
print("===== Pyradiomics 版本信息 =====")
print(f"当前版本：{radiomics.__version__}")
#print(f"是否启用C++扩展：{radiomics.getCExtensionStatus()}")
print(f"SimpleITK版本：{sitk.__version__}")
print(f"NumPy版本：{np.__version__}")

# 替换为你的病例路径（用1016这个病例测试）
CASE_ID = "1016"
ROOT_DIR = r"D:\gulianyu\LungAd_Radiomics"
PROCESSED_DIR = os.path.join(ROOT_DIR, "compilation_data")

# 1. 加载PET图像和ROI
pet_path = os.path.join(PROCESSED_DIR, CASE_ID, f"{CASE_ID}_PET_registered.nii.gz")
roi_path = os.path.join(PROCESSED_DIR, CASE_ID, f"{CASE_ID}_ROI_bin.nii.gz")

pet_img = sitk.ReadImage(pet_path)
roi_img = sitk.ReadImage(roi_path)

# 转换为numpy数组（方便分析）
pet_np = sitk.GetArrayFromImage(pet_img)
roi_np = sitk.GetArrayFromImage(roi_img)

# 2. 验证1：图像和ROI维度是否一致
print("===== 维度验证 =====")
print(f"PET图像维度：{pet_np.shape}")
print(f"ROI掩码维度：{roi_np.shape}")
if pet_np.shape != roi_np.shape:
    print("❌ 维度不匹配！这是核心问题！")
else:
    print("✅ 维度匹配")

# 3. 验证2：ROI内体素数量
roi_voxels = np.sum(roi_np > 0)
print("\n===== ROI验证 =====")
print(f"ROI内体素总数：{roi_voxels}")
if roi_voxels < 10:
    print("❌ ROI体素过少（<10），无法计算纹理特征！")
elif roi_voxels < 30:
    print("⚠️ ROI体素偏少（<30），可能无法计算部分纹理特征")
else:
    print("✅ ROI体素数量正常")

# 4. 验证3：ROI内PET值的分布
pet_roi = pet_np[roi_np > 0]
print("\n===== PET值验证 =====")
print(f"ROI内PET值范围：{pet_roi.min()} ~ {pet_roi.max()}")
print(f"ROI内PET值唯一值数量：{len(np.unique(pet_roi))}")
if len(np.unique(pet_roi)) <= 1:
    print("❌ ROI内PET值无变化，无法计算纹理特征！")
elif len(np.unique(pet_roi)) < 5:
    print("⚠️ ROI内PET值变化少，纹理特征可能被跳过")
else:
    print("✅ PET值分布正常")

# 5. 验证4：图像空间信息
print("\n===== 空间信息验证 =====")
print(f"PET体素间距：{pet_img.GetSpacing()}")
print(f"ROI体素间距：{roi_img.GetSpacing()}")
if pet_img.GetSpacing() != roi_img.GetSpacing():
    print("❌ 体素间距不匹配！")
else:
    print("✅ 体素间距匹配")