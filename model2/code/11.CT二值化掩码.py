# -*- coding: utf-8 -*-
import os
import SimpleITK as sitk
import numpy as np

# ===================== 配置全局路径（仅需修改ROOT_DIR） =====================
ROOT_DIR = r"D:\gulianyu\LungAd_Radiomics"  # 替换为你的根目录（和之前一致）
RAW_ROI_DIR = os.path.join(ROOT_DIR, "raw_roi")  # 原始CT ROI路径
BINARIZED_ROI_DIR = os.path.join(ROOT_DIR, "binarized_roi")  # 二值化后输出路径


# ===================== 核心函数：单个ROI二值化 =====================
def binarize_roi(case_id):
    """
    将原始CT ROI转换为二进制掩码（仅0和1，1为肿瘤区域）
    :param case_id: 病例编号（如case_001）
    """
    # 1. 构建输入输出路径
    raw_roi_path = os.path.join(RAW_ROI_DIR, f"{case_id}_ROI.nii.gz")
    output_roi_path = os.path.join(BINARIZED_ROI_DIR, case_id, f"{case_id}_ROI_bin.nii.gz")

    # 检查原始ROI是否存在
    if not os.path.exists(raw_roi_path):
        print(f"❌ 病例{case_id}：原始ROI缺失 → {raw_roi_path}")
        return False

    try:
        # 2. 读取原始ROI（保留空间信息：分辨率、坐标等）
        roi_image = sitk.ReadImage(raw_roi_path)
        roi_array = sitk.GetArrayFromImage(roi_image)  # 转为numpy数组

        # 3. 二值化核心逻辑：所有大于0的像素设为1，其余为0
        # （无论原始ROI是uint16/int32，统一转为0/1的8位无符号整数）
        roi_binary_array = np.where(roi_array > 0, 1, 0).astype(np.uint8)

        # 4. 转回SimpleITK图像，保留原始空间信息（关键！保证和CT/PET对齐）
        roi_binary_image = sitk.GetImageFromArray(roi_binary_array)
        roi_binary_image.CopyInformation(roi_image)  # 复制分辨率、坐标、方向

        # 5. 创建病例输出目录并保存
        os.makedirs(os.path.join(BINARIZED_ROI_DIR, case_id), exist_ok=True)
        sitk.WriteImage(roi_binary_image, output_roi_path)

        print(f"✅ 病例{case_id}：ROI二值化完成 → {output_roi_path}")
        return True

    except Exception as e:
        print(f"❌ 病例{case_id}：ROI二值化失败 → {str(e)}")
        return False


# ===================== 批量处理所有病例 =====================
def batch_binarize_all_roi():
    # 1. 获取所有原始ROI文件（提取病例编号）
    raw_roi_files = [
        f for f in os.listdir(RAW_ROI_DIR)
        if f.endswith("_ROI.nii.gz") and os.path.isfile(os.path.join(RAW_ROI_DIR, f))
    ]
    # 提取病例编号（如case_001_ROI.nii.gz → case_001）
    case_ids = [f.replace("_ROI.nii.gz", "") for f in raw_roi_files]
    case_ids.sort()  # 按编号排序

    # 2. 遍历所有病例执行二值化
    print(f"📌 开始批量ROI二值化，共{len(case_ids)}个病例\n")
    for case_id in case_ids:
        binarize_roi(case_id)

    print("\n🎉 批量ROI二值化完成！请检查 binarized_roi 文件夹。")


# ===================== 运行主函数 =====================
if __name__ == "__main__":
    batch_binarize_all_roi()