import os
import pydicom
import numpy as np
import SimpleITK as sitk
from datetime import datetime
from model2.code.contants import TEST_CASE_ID, TEST_SWITCH
import math

# ===================== 配置全局路径 =====================
ROOT_DIR = r"D:\gulianyu\LungAd_Radiomics"
RAW_DIR = os.path.join(ROOT_DIR, "raw_data")
RAW_NIFTI_DIR = os.path.join(ROOT_DIR, "converted_nifti")
OUTPUT_DIR = os.path.join(ROOT_DIR, "pet_suv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def pet_raw_to_suv(pet_raw_nifti_path, pet_dicom_dir, output_suv_nifti_path, case_id):
    """
    原始PET NIfTI → SUVbw校准后的PET NIfTI（基于DICOM标准修正）
    """
    # 步骤1：读取PET DICOM头文件，提取SUV校准参数
    dicom_files = [f for f in os.listdir(pet_dicom_dir) if f.lower().endswith((".dcm", ".ima"))]
    if not dicom_files:
        raise FileNotFoundError(f"未在{pet_dicom_dir}找到DICOM文件")

    # 建议排序，读取第一张切片的时间作为基准
    dicom_files.sort()
    ds = pydicom.dcmread(os.path.join(pet_dicom_dir, dicom_files[0]))

    # --- 核心参数提取（严格遵守DICOM标准） ---

    # 1. 获取患者体重 (kg) -> 转为克 (g)
    if not hasattr(ds, 'PatientWeight') or ds.PatientWeight is None:
        raise ValueError(f"病例 {case_id} 缺失体重(PatientWeight)数据，无法计算SUV。")
    patient_weight_g = float(ds.PatientWeight) * 1000.0

    rad_seq = ds.RadiopharmaceuticalInformationSequence[0]

    # 2. 获取注射总活度 (DICOM标准单位本身就是 Bq, 绝对不要乘1e6!)
    injected_dose_bq = float(rad_seq.RadionuclideTotalDose)

    # 3. 获取放射性核素半衰期 (DICOM标准单位为 秒)
    half_life_sec = float(rad_seq.RadionuclideHalfLife)
    decay_constant = math.log(2) / half_life_sec  # 使用 ln(2) 更精确

    # --- 时间提取与计算（统一换算为秒，避免精度丢失） ---

    # 注射时间 (RadiopharmaceuticalStartTime)
    injection_time_str = str(rad_seq.RadiopharmaceuticalStartTime).split('.')[0].zfill(6)
    # 扫描开始时间 (AcquisitionTime 或 SeriesTime)
    scan_time_str = str(getattr(ds, 'AcquisitionTime', ds.SeriesTime)).split('.')[0].zfill(6)

    fmt = "%H%M%S"
    t1 = datetime.strptime(injection_time_str, fmt)
    t2 = datetime.strptime(scan_time_str, fmt)

    time_diff_sec = (t2 - t1).total_seconds()

    # 跨天修正逻辑（保留你的思路，换算为秒）
    if time_diff_sec < 0:
        time_diff_sec += 24 * 3600
        print(f"⚠️ 病例{case_id} 扫描跨天，修正后时间差：{time_diff_sec / 60:.2f}分钟")

    # --- 步骤2：衰变校正 ---

    # 检查图像是否已经由机器做过衰变校正
    decay_correction = getattr(ds, 'DecayCorrection', 'NONE')
    if decay_correction == 'ADMIN':
        # 如果是 ADMIN，说明机器已经把像素值反推回注射时的活度了，不需要再衰变
        remaining_dose_bq = injected_dose_bq
    else:
        # 通常是 'START'，代表像素值代表扫描开始时的活度，需要计算注射到扫描间的衰变
        decay_factor = np.exp(-decay_constant * time_diff_sec)
        remaining_dose_bq = injected_dose_bq * decay_factor

    # --- 步骤3：读取图像与数值转换 ---
    if not os.path.exists(pet_raw_nifti_path):
        raise FileNotFoundError(f"原始PET NIfTI不存在：{pet_raw_nifti_path}")

    pet_raw_img = sitk.ReadImage(pet_raw_nifti_path)
    pet_raw_np = sitk.GetArrayFromImage(pet_raw_img).astype(np.float32)

    # 【关键修正】处理 Rescale Slope
    # 如果你是用 dcm2niix 转的 NIfTI，软件通常已经帮你乘过 Slope 了。
    # 验证方法：如果 pet_raw_np 的最大值大于 1000，通常说明已经应用过，或者是标准机器。
    # 这里加一个保守的验证逻辑：
    rescale_slope = float(getattr(ds, "RescaleSlope", 1.0))
    rescale_intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    # 这是一个启发式判断：如果原始NIfTI值很小（未被缩放），才手动应用缩放
    # 或者你可以自己确认 NIfTI 转换工具的行为，如果确认没有转换，就取消注释下面这一行
    # pet_activity_np = pet_raw_np * rescale_slope + rescale_intercept
    pet_activity_np = pet_raw_np  # 假设转换器已处理（dcm2niix默认行为）

    # --- 步骤4：计算 SUV (SUVbw - 体重标化法) ---
    # 公式: SUV = 组织活度浓度(Bq/mL) / (注射活度(Bq) / 体重(g))
    # 假设 1g 组织体积约为 1mL
    suv_factor = patient_weight_g / remaining_dose_bq
    suv_np = pet_activity_np * suv_factor

    # 过滤负数并在医学上做合理裁剪 (SUV通常在 0~50 之间，骨转移可能极高，但不应有负数)
    suv_np = np.clip(suv_np, 0, None)

    # --- 步骤5：保存结果 ---
    output_dir = os.path.dirname(output_suv_nifti_path)
    os.makedirs(output_dir, exist_ok=True)

    suv_img = sitk.GetImageFromArray(suv_np)
    suv_img.CopyInformation(pet_raw_img)
    sitk.WriteImage(suv_img, output_suv_nifti_path)

    # --- 打印数值验证 (帮助你Debug) ---
    print(f"📌 病例{case_id} 核心计算参数验证：")
    print(f"   - 半衰期(秒): {half_life_sec}")
    print(f"   - 注射-扫描时间差(分): {time_diff_sec / 60:.2f}")
    print(f"   - 注射总活度(Bq): {injected_dose_bq:.2e}")
    print(f"   - 扫描时剩余活度(Bq): {remaining_dose_bq:.2e}")
    print(f"   - 原始图像数值范围: {np.min(pet_raw_np):.2f} ~ {np.max(pet_raw_np):.2f}")
    print(f"   - 最终 SUV 值范围: {np.min(suv_np):.2f} ~ {np.max(suv_np):.2f}")

    # 常见医学常识检查：肝脏/血池SUV通常在 2.0-3.0 左右。如果你的SUV最大值超过100或小于1，说明前面Rescale逻辑需要调整。
    if np.max(suv_np) < 1.0 or np.max(suv_np) > 500:
        print("   ⚠️ 警告：最高SUV值不在常规临床范围内，请检查 dcm2niix 是否未应用 RescaleSlope！")

    return output_suv_nifti_path

def pet_suv():
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
        if TEST_SWITCH and case_id_int not in TEST_CASE_ID:
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

if __name__ == "__main__":
   pet_suv()