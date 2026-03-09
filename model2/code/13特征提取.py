# -*- coding: utf-8 -*-
import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import warnings

warnings.filterwarnings('ignore')  # 屏蔽无关警告

# ===================== 配置全局路径（仅需修改ROOT_DIR） =====================
ROOT_DIR = r"D:\gulianyu\LungAd_Radiomics"  # 替换为你的根目录
PROCESSED_DIR = os.path.join(ROOT_DIR, "compilation_data")  # 预处理后的数据目录
OUTPUT_DIR = os.path.join(ROOT_DIR, "features_output")  # 特征输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== 1. 配置特征提取参数（适配pyradiomics 2.x，肺癌研究标准） =====================
def get_lung_radiomics_extractor():
    """
    配置肺癌影像组学特征提取器（适配pyradiomics 2.x，兼容所有旧版本）
    """
    # 初始化提取器
    extractor = featureextractor.RadiomicsFeatureExtractor()

    # ---------------------- 核心参数设置（肺癌研究重点） ----------------------
    # 1. 影像重采样（统一体素分辨率，避免设备差异）
    extractor.settings['resampledPixelSpacing'] = [1, 1, 1]  # 重采样为1×1×1mm（肺癌标准）
    extractor.settings['interpolator'] = sitk.sitkBSpline  # 插值方式（避免边缘失真）

    # 2. 特征类型选择（旧版本逻辑：先禁用所有，再启用需要的）
    extractor.disableAllFeatures()  # 第一步：禁用所有特征（旧版本核心）
    # 启用肺癌研究常用特征（仅这5类，其余默认禁用）
    extractor.enableFeatureClassByName('firstorder')  # 一阶统计特征（均值、方差、熵）
    extractor.enableFeatureClassByName('glcm')  # 灰度共生矩阵（纹理核心）
    extractor.enableFeatureClassByName('glrlm')  # 灰度游程矩阵（异质性）
    extractor.enableFeatureClassByName('glszm')  # 灰度大小区域矩阵（区域分布）
    extractor.enableFeatureClassByName('ngtdm')  # 邻域灰度差矩阵（局部纹理）

    # 3. ROI掩码设置（二值化掩码专用）
    extractor.settings['maskedKernel'] = True  # 仅计算ROI内的特征
    extractor.settings['minimumROISize'] = 10  # 最小ROI体素数（过滤小噪声）

    return extractor


# ===================== 2. 单病例特征提取 =====================
def extract_single_case(case_id, extractor, modality='CT'):
    """
    提取单个病例的特征
    :param case_id: 病例编号（如case_001）
    :param extractor: 特征提取器
    :param modality: 模态（CT/PET）
    :return: 特征字典（含病例ID）
    """
    # 构建文件路径
    img_path = os.path.join(PROCESSED_DIR, case_id, f"{case_id}_{modality}.nii.gz")
    if modality == 'PET':
        img_path = os.path.join(PROCESSED_DIR, case_id, f"{case_id}_{modality}_registered.nii.gz")
    roi_path = os.path.join(PROCESSED_DIR, case_id, f"{case_id}_ROI_bin.nii.gz")

    # 检查文件是否存在
    if not os.path.exists(img_path):
        print(f"❌ 病例{case_id}：{modality}文件缺失 → {img_path}")
        return None
    if not os.path.exists(roi_path):
        print(f"❌ 病例{case_id}：ROI文件缺失 → {roi_path}")
        return None

    try:
        # 提取特征（返回字典：{特征名: 特征值}）
        features = extractor.execute(img_path, roi_path)
        # 整理特征：仅保留数值型特征（过滤元信息）
        feature_dict = {'case_id': case_id}  # 新增病例ID列（建模关键）
        for key, value in features.items():
            # 仅保留特征值（排除diagnostics开头的元信息）
            if not key.startswith('diagnostics_'):
                feature_dict[key] = value
        print(f"✅ 病例{case_id}：{modality}特征提取完成")
        return feature_dict
    except Exception as e:
        print(f"❌ 病例{case_id}：{modality}特征提取失败 → {str(e)}")
        return None


# ===================== 3. 批量提取所有病例特征 =====================
def batch_extract_features():
    # 初始化提取器（适配旧版本）
    extractor = get_lung_radiomics_extractor()

    # 获取所有病例文件夹
    case_folders = [
        folder for folder in os.listdir(PROCESSED_DIR)
        if os.path.isdir(os.path.join(PROCESSED_DIR, folder))
    ]
    case_folders.sort()  # 按编号排序
    print(f"📌 开始批量特征提取，共{len(case_folders)}个病例\n")

    # ---------------------- 提取CT特征 ----------------------
    # ct_features_list = []
    # print("========== 提取CT特征 ==========")
    # for case_id in case_folders:
    #     ct_feat = extract_single_case(case_id, extractor, modality='CT')
    #     if ct_feat:
    #         ct_features_list.append(ct_feat)
    #
    # # 保存CT特征表
    # ct_df = pd.DataFrame(ct_features_list)
    # ct_df.to_csv(os.path.join(OUTPUT_DIR, "ct_features.csv"), index=False, encoding='utf-8')
    # print(f"\n✅ CT特征表保存完成 → {os.path.join(OUTPUT_DIR, 'ct_features.csv')}")

    # ---------------------- 提取PET特征 ----------------------
    pet_features_list = []
    print("\n========== 提取PET特征 ==========")
    for case_id in case_folders:
        pet_feat = extract_single_case(case_id, extractor, modality='PET')
        if pet_feat:
            pet_features_list.append(pet_feat)

    # 保存PET特征表
    pet_df = pd.DataFrame(pet_features_list)
    pet_df.to_csv(os.path.join(OUTPUT_DIR, "pet_features.csv"), index=False, encoding='utf-8')
    print(f"\n✅ PET特征表保存完成 → {os.path.join(OUTPUT_DIR, 'pet_features.csv')}")

    # # ---------------------- 合并CT+PET特征 ----------------------
    # if not ct_df.empty and not pet_df.empty:
    #     combined_df = pd.merge(ct_df, pet_df, on='case_id', suffixes=('_CT', '_PET'))
    #     combined_df.to_csv(os.path.join(OUTPUT_DIR, "combined_features.csv"), index=False, encoding='utf-8')
    #     print(f"\n✅ 合并特征表保存完成 → {os.path.join(OUTPUT_DIR, 'combined_features.csv')}")
    # else:
    #     print("\n⚠️  CT/PET特征表为空，未生成合并表")

    print("\n🎉 所有病例特征提取完成！")


# ===================== 运行主函数 =====================
if __name__ == "__main__":
    batch_extract_features()