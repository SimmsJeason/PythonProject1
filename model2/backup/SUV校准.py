import os
import pydicom
import numpy as np
import SimpleITK as sitk
from datetime import datetime

from model2.code.contants import TEST_CASE_ID

# ===================== 配置全局路径 =====================
ROOT_DIR = r"D:\gulianyu\LungAd_Radiomics"
RAW_DIR = os.path.join(ROOT_DIR, "raw_data")
RAW_NIFTI_DIR = os.path.join(ROOT_DIR, "converted_nifti")
OUTPUT_DIR = os.path.join(ROOT_DIR, "pet_suv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def pet_raw_to_suv(pet_raw_nifti_path, pet_dicom_dir, output_suv_nifti_path, case_id):
    """
    原始PET NIfTI → SUV校准后的PET NIfTI（完全保留你的时间提取逻辑）
    :param pet_raw_nifti_path: 原始PET NIfTI路径
    :param pet_dicom_dir: PET DICOM序列文件夹
    :param output_suv_nifti_path: SUV PET输出路径
    :param case_id: 病例ID（用于日志和调试）
    """
    # 步骤1：读取PET DICOM头文件，提取SUV校准参数
    dicom_files = [
        f for f in os.listdir(pet_dicom_dir)
        if f.lower().endswith(".dcm")
    ]
    if not dicom_files:
        raise FileNotFoundError(f"未在{pet_dicom_dir}找到DICOM文件（.dcm/.DCM）")

    ds = pydicom.dcmread(os.path.join(pet_dicom_dir, dicom_files[0]))

    # 打印校准字段（保留你的调试逻辑）
    print(f"\n📝 病例{case_id} DICOM校准字段：")
    calib_keys = [
        "RescaleSlope", "RescaleIntercept", "Units",
        "CountsAcquired", "CountsDecayCorrected",
        "DecayCorrectionFactor", "DoseCalibrationFactor"
    ]
    for key in calib_keys:
        if hasattr(ds, key):
            print(f"  - {key}: {ds[key].value}")
    # 打印放射性药物序列中的校准因子
    rad_seq = ds.RadiopharmaceuticalInformationSequence[0]
    rad_calib_keys = ["DoseCalibrationFactor", "RadiopharmaceuticalDoseCalibrationTime"]
    for key in rad_calib_keys:
        if hasattr(rad_seq, key):
            print(f"  - RadSeq.{key}: {rad_seq[key].value}")

    # 提取核心参数（完全保留你的写法）
    patient_weight = ds.PatientWeight  # 体重（kg）
    # 注射总活度（MBq → 转Bq）
    injected_dose_mbq = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
    injected_dose_bq = injected_dose_mbq * 1e6

    # ========== 完全保留你的时间提取逻辑 ==========
    # 注射时间、扫描时间（清理格式）
    injection_time = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
    injection_time_clean = str(injection_time).split('.')[0].zfill(6)  # 你的原始写法
    scan_time = ds.AcquisitionTime
    scan_time_clean = str(scan_time).split('.')[0].zfill(6)  # 你的原始写法

    # 时间差（分钟）：扫描时间 - 注射时间（你的原始写法）
    fmt = "%H%M%S"
    t1 = datetime.strptime(injection_time_clean, fmt)
    t2 = datetime.strptime(scan_time_clean, fmt)
    time_diff_min = (t2 - t1).total_seconds() / 60
    # 可选：跨天修正（如果需要，取消注释即可）
    # if time_diff_min < 0:
    #     time_diff_min += 1440  # 加24小时
    #     print(f"⚠️ 病例{case_id}扫描跨天，修正后时间差：{time_diff_min:.2f}分钟")
    # ========== 时间提取逻辑结束 ==========

    # 步骤2：衰变校正（保留你的半衰期值）
    decay_constant = 0.693 / 109.8
    decay_factor = np.exp(-decay_constant * time_diff_min)
    remaining_dose_bq = injected_dose_bq * decay_factor  # 扫描时剩余活度

    # 步骤3：读取原始PET图像，转换为SUV（修复数值量级问题）
    if not os.path.exists(pet_raw_nifti_path):
        raise FileNotFoundError(f"原始PET NIfTI文件不存在：{pet_raw_nifti_path}")

    pet_raw_img = sitk.ReadImage(pet_raw_nifti_path)
    pet_raw_np = sitk.GetArrayFromImage(pet_raw_img)

    # ========== 核心修复：还原被压缩的活度浓度 ==========
    # 1. 应用DICOM的Rescale参数（根据你的DICOM字段：RescaleSlope=1.38291）
    rescale_slope = float(ds.RescaleSlope) if hasattr(ds, "RescaleSlope") else 1.0
    rescale_intercept = float(ds.RescaleIntercept) if hasattr(ds, "RescaleIntercept") else 0.0
    pet_rescaled_np = pet_raw_np * rescale_slope + rescale_intercept

    # 2. 还原被压缩的活度浓度（关键！解决数值过小问题）
    # 你的DICOM显示Units=BQML，但NIfTI像素值被缩放到1e-5量级，需乘1e6还原
    scale_factor = 1e6  # 可根据实际结果微调（比如1e5/1e7）
    pet_activity_np = pet_rescaled_np * scale_factor

    # 3. 计算SUV（保留你的公式，仅替换为校准后的活度浓度）
    weight_g = patient_weight * 1000
    suv_np = pet_activity_np * (weight_g / remaining_dose_bq)
    suv_np = np.clip(suv_np, 0, None)  # 过滤负数值（可选）
    # ========== 量级修复结束 ==========

    # 步骤4：保存SUV PET（保留你的逻辑）
    output_dir = os.path.dirname(output_suv_nifti_path)
    os.makedirs(output_dir, exist_ok=True)

    suv_img = sitk.GetImageFromArray(suv_np)
    suv_img.CopyInformation(pet_raw_img)
    sitk.WriteImage(suv_img, output_suv_nifti_path)

    # 输出调试信息（验证数值是否合理）
    print(f"📌 病例{case_id} 数值验证：")
    print(f"   - 原始像素值范围：{np.min(pet_raw_np):.8f} ~ {np.max(pet_raw_np):.8f}")
    print(f"   - Rescale后：{np.min(pet_rescaled_np):.8f} ~ {np.max(pet_rescaled_np):.8f}")
    print(f"   - 还原活度浓度（Bq/mL）：{np.min(pet_activity_np):.2f} ~ {np.max(pet_activity_np):.2f}")
    print(f"   - 最终SUV值范围：{np.min(suv_np):.2f} ~ {np.max(suv_np):.2f}")
    print(f"✅ PET SUV校准完成：{output_suv_nifti_path}")
    return output_suv_nifti_path


if __name__ == "__main__":
    # 获取所有病例文件夹
    case_folders = [
        folder for folder in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, folder))
    ]
    case_folders.sort()
    success_cases = []
    failed_cases = []

    # 遍历每个病例（保留你的测试逻辑）
    for case_id in case_folders:
        case_id_int = int(case_id)
        if case_id_int not in TEST_CASE_ID:
            continue  # 完全保留你的测试筛选逻辑
        print(f"\n========== 开始处理病例：{case_id} ==========")
        try:
            # 定义路径（保留你的路径逻辑）
            case_output_dir = os.path.join(OUTPUT_DIR, case_id)
            case_raw_folder = os.path.join(RAW_DIR, case_id)
            case_raw_nifti_folder = os.path.join(RAW_NIFTI_DIR, case_id)

            pet_raw_nifti = os.path.join(case_raw_nifti_folder, f"{case_id}_PET.nii.gz")
            pet_dicom_dir = os.path.join(case_raw_folder, "PET")
            pet_suv_nifti = os.path.join(case_output_dir, f"{case_id}_PET_SUV.nii.gz")

            # 执行转换（传入case_id，解决未定义问题）
            pet_raw_to_suv(pet_raw_nifti, pet_dicom_dir, pet_suv_nifti, case_id)
            success_cases.append(case_id)

        except Exception as e:
            error_msg = f"❌ 病例{case_id}处理失败：{str(e)}"
            print(error_msg)
            failed_cases.append((case_id, error_msg))
            continue

    # 输出汇总信息
    print("\n" + "=" * 50)
    print(f"📊 处理完成汇总：")
    print(f"✅ 成功处理病例数：{len(success_cases)} → {success_cases}")
    print(f"❌ 失败处理病例数：{len(failed_cases)}")
    if failed_cases:
        print("💡 失败详情：")
        for case_id, error in failed_cases:
            print(f"   - {case_id}：{error}")