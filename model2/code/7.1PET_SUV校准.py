import os
import pydicom
import numpy as np
import SimpleITK as sitk
from datetime import datetime  # 提前导入，避免函数内重复导入

# ===================== 配置全局路径（仅需修改ROOT_DIR） =====================
ROOT_DIR = r"D:\gulianyu\LungAd_Radiomics"  # 替换为你的根目录
RAW_DIR = os.path.join(ROOT_DIR, "raw_data")  # 原始数据目录
RAW_NIFTI_DIR = os.path.join(ROOT_DIR, "converted_nifti")  # 转换后的数据目录
OUTPUT_DIR = os.path.join(ROOT_DIR, "pet_suv")  # 特征输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


def pet_raw_to_suv(pet_raw_nifti_path, pet_dicom_dir, output_suv_nifti_path):
    """
    原始PET NIfTI → SUV校准后的PET NIfTI
    :param pet_raw_nifti_path: 原始PET NIfTI路径
    :param pet_dicom_dir: PET DICOM序列文件夹（读取头文件参数）
    :param output_suv_nifti_path: SUV PET输出路径
    """
    # 步骤1：读取PET DICOM头文件，提取SUV校准参数
    # 读取第一个DICOM文件（头文件信息一致）- 兼容.dcm和.DCM后缀
    dicom_files = [
        f for f in os.listdir(pet_dicom_dir)
        if f.lower().endswith(".dcm")  # 转小写后判断，兼容大小写
    ]
    if not dicom_files:
        raise FileNotFoundError(f"未在{pet_dicom_dir}找到DICOM文件（.dcm/.DCM）")

    ds = pydicom.dcmread(os.path.join(pet_dicom_dir, dicom_files[0]))

    # 提取核心参数（根据DICOM字段适配，不同设备字段名可能略有差异）
    patient_weight = ds.PatientWeight  # 体重（kg）
    # 注射总活度（MBq → 转Bq）
    injected_dose_mbq = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
    injected_dose_bq = injected_dose_mbq * 1e6

    # 注射时间、扫描时间（清理格式）
    injection_time = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
    injection_time_clean = str(injection_time).split('.')[0].zfill(6)  # 分割小数+补零到6位
    scan_time = ds.AcquisitionTime
    scan_time_clean = str(scan_time).split('.')[0].zfill(6)

    # 时间差（分钟）：扫描时间 - 注射时间
    fmt = "%H%M%S"  # DICOM时间格式：HHMMSS
    t1 = datetime.strptime(injection_time_clean, fmt)
    t2 = datetime.strptime(scan_time_clean, fmt)
    time_diff_min = (t2 - t1).total_seconds() / 60

    # 步骤2：衰变校正（¹⁸F半衰期109.8分钟）
    decay_constant = 0.693 / 109.8
    decay_factor = np.exp(-decay_constant * time_diff_min)
    remaining_dose_bq = injected_dose_bq * decay_factor  # 扫描时剩余活度

    # 步骤3：读取原始PET图像，转换为SUV
    if not os.path.exists(pet_raw_nifti_path):
        raise FileNotFoundError(f"原始PET NIfTI文件不存在：{pet_raw_nifti_path}")

    pet_raw_img = sitk.ReadImage(pet_raw_nifti_path)
    pet_raw_np = sitk.GetArrayFromImage(pet_raw_img)  # 原始计数（counts）
    # SUV公式：SUV = (像素活度浓度 × 体重(g)) / 剩余活度(Bq)
    # 像素活度浓度 = 原始计数 × 设备校准因子（这里简化：pet_raw_np已为活度浓度 Bq/mL）
    weight_g = patient_weight * 1000  # 体重转克
    suv_np = pet_raw_np * (weight_g / remaining_dose_bq) * 1000  # 单位转换

    # 步骤4：保存SUV PET（保留原始空间信息）
    # 确保输出目录存在（关键修复：提前创建输出文件所在的目录）
    output_dir = os.path.dirname(output_suv_nifti_path)
    os.makedirs(output_dir, exist_ok=True)

    suv_img = sitk.GetImageFromArray(suv_np)
    suv_img.CopyInformation(pet_raw_img)  # 继承原始PET的空间参数
    sitk.WriteImage(suv_img, output_suv_nifti_path)
    print(f"✅ PET SUV校准完成：{output_suv_nifti_path}")
    return output_suv_nifti_path


if __name__ == "__main__":
    # 获取所有病例文件夹（仅保留目录）
    case_folders = [
        folder for folder in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, folder))
    ]
    case_folders.sort()

    # 记录成功/失败的病例
    success_cases = []
    failed_cases = []

    # 遍历每个病例，转换PET到SUV
    for case_id in case_folders:
        print(f"\n========== 开始处理病例：{case_id} ==========")
        try:
            # 1. 定义路径（关键修复：避免路径重复拼接case_id）
            case_output_dir = os.path.join(OUTPUT_DIR, case_id)  # 病例输出目录：pet_suv/1016
            case_raw_folder = os.path.join(RAW_DIR, case_id)  # 原始DICOM目录：raw_data/1016
            case_raw_nifti_folder = os.path.join(RAW_NIFTI_DIR, case_id)  # 原始NIfTI目录：converted_nifti/1016

            # 2. 拼接文件路径（修复后路径：pet_suv/1016/1016_PET_SUV.nii.gz）
            pet_raw_nifti = os.path.join(case_raw_nifti_folder, f"{case_id}_PET.nii.gz")
            pet_dicom_dir = os.path.join(case_raw_folder, "PET")
            pet_suv_nifti = os.path.join(case_output_dir, f"{case_id}_PET_SUV.nii.gz")

            # 3. 执行SUV转换
            pet_raw_to_suv(pet_raw_nifti, pet_dicom_dir, pet_suv_nifti)
            success_cases.append(case_id)

        # 捕获所有异常，仅打印错误信息，继续执行下一个病例
        except Exception as e:
            error_msg = f"❌ 病例{case_id}处理失败：{str(e)}"
            print(error_msg)
            failed_cases.append((case_id, error_msg))
            continue  # 跳过当前病例，处理下一个

    # 处理完成后输出汇总信息
    print("\n" + "=" * 50)
    print(f"📊 处理完成汇总：")
    print(f"✅ 成功处理病例数：{len(success_cases)} → {success_cases}")
    print(f"❌ 失败处理病例数：{len(failed_cases)}")
    if failed_cases:
        print("💡 失败详情：")
        for case_id, error in failed_cases:
            print(f"   - {case_id}：{error}")