# -*- coding: utf-8 -*-
import os
import SimpleITK as sitk
import numpy as np
import sys

# ===================== 配置全局路径（唯一需要修改的地方） =====================
# 替换为你的根目录绝对路径（路径用r""包裹，避免转义符问题）
ROOT_DIR = r"D:\gulianyu\LungAd_Radiomics"
# 原始DICOM存放目录（无需修改，按文件夹结构自动拼接）
RAW_DICOM_DIR = os.path.join(ROOT_DIR, "raw_data")
# 转换后NIfTI输出目录（无需修改）
CONVERTED_DIR = os.path.join(ROOT_DIR, "converted_nifti")

ERROR_ID = [1016,1260,2198,2222,2223,2224,2225,2226,2228,2232,2237,2238,2239,2243,2247,2248,2253,2255,2257,2258,2259,2260,2262,2263,2266,2269,2271,2278,2282,2283,2298,2301,2304,2307,2308,2310]

# ===================== 核心函数：单个DICOM序列转NIfTI =====================
def convert_dicom_series_to_nifti(dicom_series_dir, output_nifti_path):
    """
    将单个DICOM序列（如CT/PET）转换为NIfTI格式（.nii.gz）
    :param dicom_series_dir: DICOM序列文件夹路径（如case_001/CT）
    :param output_nifti_path: 输出NIfTI文件路径（如converted_nifti/case_001/case_001_CT.nii.gz）
    """
    try:
        # 1. 读取DICOM序列（自动识别、排序、拼接为3D图像）
        reader = sitk.ImageSeriesReader()
        # 获取DICOM文件夹中的所有切片文件名（按扫描顺序排序）
        dicom_file_names = reader.GetGDCMSeriesFileNames(dicom_series_dir)

        # 检查是否找到DICOM文件
        if len(dicom_file_names) == 0:
            print(f"❌ 错误：{dicom_series_dir} 中未找到DICOM文件！")
            return False

        reader.SetFileNames(dicom_file_names)
        # 读取并拼接为3D图像
        dicom_image = reader.Execute()

        # 2. 保存为NIfTI格式（.nii.gz，压缩格式，节省空间）
        sitk.WriteImage(dicom_image, output_nifti_path)
        print(f"✅ 转换成功：{output_nifti_path}")
        return True

    except Exception as e:
        # 捕获错误并打印，避免单个病例出错导致整个脚本中断
        print(f"❌ 转换失败 {dicom_series_dir}：{str(e)}")
        return False


# ===================== 批量处理所有病例 =====================
def batch_convert_all_cases():
    # 1. 创建输出目录（如果不存在）
    os.makedirs(CONVERTED_DIR, exist_ok=True)

    # 2. 获取所有病例文件夹（如case_001、case_002）
    case_folders = [
        folder for folder in os.listdir(RAW_DICOM_DIR)
        if os.path.isdir(os.path.join(RAW_DICOM_DIR, folder))
    ]
    # 按病例编号排序（保证处理顺序和编号一致）
    case_folders.sort()

    # 3. 遍历每个病例，转换CT和PET
    for case_id in case_folders:
        #重新执行失败的id
        if case_id not in ERROR_ID:
            continue
        print(f"\n========== 开始处理病例：{case_id} ==========")

        # 创建该病例的输出子文件夹
        case_output_dir = os.path.join(CONVERTED_DIR, case_id)
        os.makedirs(case_output_dir, exist_ok=True)

        # ---------------------- 转换CT ----------------------
        ct_dicom_dir = os.path.join(RAW_DICOM_DIR, case_id, "CT")
        ct_nifti_path = os.path.join(case_output_dir, f"{case_id}_CT.nii.gz")
        if os.path.exists(ct_dicom_dir):
            convert_dicom_series_to_nifti(ct_dicom_dir, ct_nifti_path)
        else:
            print(f"❌ 未找到CT DICOM文件夹：{ct_dicom_dir}")

        # ---------------------- 转换PET ----------------------
        pet_dicom_dir = os.path.join(RAW_DICOM_DIR, case_id, "PET")
        pet_nifti_path = os.path.join(case_output_dir, f"{case_id}_PET.nii.gz")
        if os.path.exists(pet_dicom_dir):
            convert_dicom_series_to_nifti(pet_dicom_dir, pet_nifti_path)
        else:
            print(f"❌ 未找到PET DICOM文件夹：{pet_dicom_dir}")

    print("\n🎉 所有病例处理完成！请检查 converted_nifti 文件夹。")


# ===================== 运行主函数 =====================
if __name__ == "__main__":
    batch_convert_all_cases()