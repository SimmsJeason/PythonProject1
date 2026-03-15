# -*- coding: utf-8 -*-
import os
import numpy as np
import SimpleITK as sitk

# ===================== 配置（请根据你的实际路径修改）=====================
ROOT_DIR = r"D:\gulianyu\LungAd_Radiomics"
PROCESSED_DIR = os.path.join(ROOT_DIR, "compilation_data")
MIN_ROI_VOXELS = 10  # 最小有效ROI体素数


def validate_case_data(case_id, modality='CT'):
    """
    校验指定病例的图像和ROI数据有效性
    :param case_id: 病例ID（字符串/数字）
    :param modality: 模态，可选 'CT' 或 'PET'
    :return: None（直接打印校验结论）
    """
    case_id = str(case_id)
    modality = modality.upper()
    print(f"\n{'=' * 50}")
    print(f"开始校验病例 {case_id} - {modality} 数据")
    print('=' * 50)

    # 1. 构建文件路径
    if modality == 'CT':
        img_path = os.path.join(PROCESSED_DIR, case_id, f"{case_id}_CT.nii.gz")
    else:
        img_path = os.path.join(PROCESSED_DIR, case_id, f"{case_id}_PET_registered.nii.gz")
    roi_path = os.path.join(PROCESSED_DIR, case_id, f"{case_id}_ROI_bin.nii.gz")

    # 2. 检查文件是否存在
    if not os.path.exists(img_path):
        print(f"❌ 错误：{modality}图像文件不存在")
        print(f"   文件路径：{img_path}")
        return
    if not os.path.exists(roi_path):
        print(f"❌ 错误：ROI文件不存在")
        print(f"   文件路径：{roi_path}")
        return
    print("✅ 文件存在性检查：通过")

    # 3. 读取图像和ROI数据
    try:
        img = sitk.ReadImage(img_path)
        roi = sitk.ReadImage(roi_path)
        img_arr = sitk.GetArrayFromImage(img)
        roi_arr = sitk.GetArrayFromImage(roi)
        print("✅ 图像/ROI读取：成功")
    except Exception as e:
        print(f"❌ 错误：读取图像/ROI失败 - {str(e)}")
        return

    # 4. 检查尺寸一致性
    if img.GetSize() != roi.GetSize():
        print(f"❌ 错误：图像与ROI尺寸不匹配")
        print(f"   图像尺寸：{img.GetSize()} | ROI尺寸：{roi.GetSize()}")
        return
    print("✅ 尺寸一致性检查：通过")

    # 5. 分析ROI数据
    roi_voxels = np.sum(roi_arr > 0)
    roi_unique_vals = np.unique(roi_arr)
    print(f"\n📊 ROI分析：")
    print(f"   ROI非零体素数：{roi_voxels} (最小要求：{MIN_ROI_VOXELS})")
    print(f"   ROI包含的像素值：{roi_unique_vals}")

    if roi_voxels < MIN_ROI_VOXELS:
        print(f"❌ 警告：ROI体素数不足，无法提取有效特征")
    else:
        print(f"✅ ROI体素数检查：通过")

    # 6. 分析图像灰度值（重点）
    print(f"\n📊 {modality}图像灰度值分析：")
    print(f"   全局灰度范围：[{img_arr.min():.2f}, {img_arr.max():.2f}]")
    print(f"   全局灰度均值：{img_arr.mean():.2f}")

    # 7. 分析ROI覆盖区域的图像灰度值（核心）
    if roi_voxels > 0:
        roi_mask = roi_arr > 0
        roi_img_vals = img_arr[roi_mask]
        roi_min = roi_img_vals.min()
        roi_max = roi_img_vals.max()
        roi_mean = roi_img_vals.mean()
        zero_ratio = (roi_img_vals == 0).sum() / len(roi_img_vals) * 100

        print(f"\n📊 ROI覆盖区域{modality}灰度值分析：")
        print(f"   ROI内灰度范围：[{roi_min:.2f}, {roi_max:.2f}]")
        print(f"   ROI内灰度均值：{roi_mean:.2f}")
        print(f"   ROI内值为0的像素占比：{zero_ratio:.2f}%")

        # 关键结论判断
        print(f"\n🎯 核心校验结论：")
        if zero_ratio == 100:
            print(f"   ❌ 严重问题：ROI内所有{modality}灰度值均为0 → 非形状特征全为0")
        elif modality == 'CT' and (roi_min == roi_max == 0):
            print(f"   ❌ 严重问题：ROI内CT值恒为0 → 非形状特征全为0")
        elif modality == 'CT' and (roi_min < -1000 or roi_max > 3000):
            print(f"   ⚠️  警告：CT值超出正常范围（-1000~1000），可能是数据格式错误")
        elif roi_mean == 0:
            print(f"   ❌ 严重问题：ROI内{modality}灰度均值为0 → 非形状特征大概率为0")
        else:
            print(f"   ✅ 正常：ROI内{modality}灰度值分布合理，特征提取应正常")
    else:
        print(f"\n🎯 核心校验结论：")
        print(f"   ❌ 严重问题：ROI为空，无法提取任何特征")

    # 8. 提取器配置建议
    print(f"\n💡 配置建议：")
    if 255 in roi_unique_vals:
        print(f"   - 提取器maskPixelValue应设置为255（当前ROI值包含255）")
    elif 1 not in roi_unique_vals and 0 in roi_unique_vals:
        print(f"   - ROI仅包含0值，是无效掩码")
    if modality == 'CT' and roi_img_vals.mean() == 0:
        print(f"   - 检查CT图像是否被错误归一化/截断")


# ===================== 调用示例 =====================
if __name__ == "__main__":
    # 使用方法：修改case_id为你要校验的病例ID
    validate_case_data(case_id=1003, modality='CT')  # 校验CT
    # validate_case_data(case_id=1144, modality='PET')  # 如需校验PET，取消注释