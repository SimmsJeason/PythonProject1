# -*- coding: utf-8 -*-
import os
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
import traceback

# ===================== 配置路径 =====================
ROOT_DIR = r"D:\gulianyu\LungAd_Radiomics"
PROCESSED_DIR = os.path.join(ROOT_DIR, "compilation_data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "features_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_CASE = "1016"

# ===================== 提取器工厂（可配置特征类）=====================
def get_extractor(enable_firstorder=True, enable_shape=True,
                  enable_glcm=False, enable_glrlm=False,
                  enable_glszm=False, enable_ngtdm=False,
                  enable_gldm=False,
                  binWidth=None, binCount=None):
    """
    灵活配置提取器，可选择启用哪些特征类，并设置离散化参数。
    """
    extractor = featureextractor.RadiomicsFeatureExtractor(
        enableCExtensions=True,
        threads=1
    )

    # 重采样（必须，否则图像可能过大）
    extractor.settings['resampledPixelSpacing'] = [2, 2, 2]
    extractor.settings['interpolator'] = sitk.sitkLinear
    extractor.settings['resampleInterpolator'] = sitk.sitkLinear

    # 归一化先关闭（若后续需要再开启）
    # extractor.settings['normalize'] = True
    # extractor.settings['normalizeScale'] = 1000

    # 离散化设置：优先使用 binCount，如果指定的话
    if binCount is not None:
        extractor.settings['binCount'] = binCount
    elif binWidth is not None:
        extractor.settings['binWidth'] = binWidth
    else:
        # 默认 binWidth=5.0
        extractor.settings['binWidth'] = 5.0

    # 其他设置
    extractor.settings['maskedKernel'] = True
    extractor.settings['minimumROISize'] = 10
    extractor.settings['maskPixelValue'] = 1

    # 特征类启用
    extractor.disableAllFeatures()
    if enable_firstorder:
        extractor.enableFeatureClassByName('firstorder')
    if enable_shape:
        extractor.enableFeatureClassByName('shape')
    if enable_glcm:
        extractor.enableFeatureClassByName('glcm')
        # 可选：限制GLCM计算量
        try:
            extractor.settings['glcm'] = {'distance': [1]}
            extractor.settings['glcm']['angles'] = [0, np.pi/2]
        except:
            pass
    if enable_glrlm:
        extractor.enableFeatureClassByName('glrlm')
    if enable_glszm:
        extractor.enableFeatureClassByName('glszm')
    if enable_ngtdm:
        extractor.enableFeatureClassByName('ngtdm')
    if enable_gldm:
        extractor.enableFeatureClassByName('gldm')

    return extractor

# ===================== 测试函数 =====================
def test_case(case_id, modality='PET', **kwargs):
    print(f"\n===== 开始测试病例 {case_id} =====")
    print("特征类启用状态：")
    for k, v in kwargs.items():
        if k.startswith('enable_'):
            print(f"  {k}: {v}")

    extractor = get_extractor(**kwargs)

    # 构建路径
    if modality == 'PET':
        img_path = os.path.join(PROCESSED_DIR, case_id, f"{case_id}_PET_registered.nii.gz")
    else:
        img_path = os.path.join(PROCESSED_DIR, case_id, f"{case_id}_{modality}.nii.gz")
    roi_path = os.path.join(PROCESSED_DIR, case_id, f"{case_id}_ROI_bin.nii.gz")

    print(f"图像路径：{img_path}")
    print(f"ROI路径：{roi_path}")

    if not os.path.exists(img_path) or not os.path.exists(roi_path):
        print("❌ 文件缺失")
        return

    # 提取前打印图像和ROI基本信息（原始）
    img_sitk = sitk.ReadImage(img_path)
    roi_sitk = sitk.ReadImage(roi_path)
    print(f"原始图像尺寸：{img_sitk.GetSize()}，间距：{img_sitk.GetSpacing()}")
    print(f"原始ROI尺寸：{roi_sitk.GetSize()}，间距：{roi_sitk.GetSpacing()}")

    # 执行提取
    try:
        features = extractor.execute(img_path, roi_path)
    except MemoryError:
        print("❌ 内存不足！尝试进一步降低灰度级数或仅启用一阶/形状特征。")
        traceback.print_exc()
        return
    except Exception as e:
        print(f"❌ 执行过程中抛出异常：{e}")
        traceback.print_exc()
        return

    # 打印诊断信息
    print("\n===== 诊断信息 =====")
    diag_keys = [k for k in features.keys() if k.startswith('diagnostics_')]
    for k in diag_keys:
        print(f"  {k} = {features[k]}")

    # 按类别统计特征数
    shape_keys = [k for k in features.keys() if k.startswith('original_shape')]
    firstorder_keys = [k for k in features.keys() if k.startswith('original_firstorder')]
    texture_keys = [k for k in features.keys() if k.startswith(('original_glcm', 'original_glrlm', 'original_glszm', 'original_ngtdm', 'original_gldm'))]
    print(f"\n形状特征：{len(shape_keys)} 项")
    print(f"一阶特征：{len(firstorder_keys)} 项")
    print(f"纹理特征：{len(texture_keys)} 项")

    total_features = len([k for k in features.keys() if not k.startswith('diagnostics_')])
    print(f"\n✅ 共提取 {total_features} 个非诊断特征")

    # 保存特征（修改后）
    import pandas as pd
    feat_dict = {'case_id': case_id}
    for k, v in features.items():
        if not k.startswith('diagnostics_'):
            # 直接保存，pandas 会处理 numpy 类型
            feat_dict[k] = v
    df = pd.DataFrame([feat_dict])
    output_path = os.path.join(OUTPUT_DIR, f"{case_id}_debug.csv")
    df.to_csv(output_path, index=False)
    print(f"特征已保存至：{output_path}，共 {len(feat_dict)-1} 个特征（不含 case_id）")

    return features

# ===================== 主流程 =====================
if __name__ == "__main__":
    # 第一步：只启用一阶和形状特征，验证基本流程
    print("【第一步】只提取一阶和形状特征")
    test_case(TEST_CASE, 'PET',
              enable_firstorder=True,
              enable_shape=True,
              enable_glcm=False,
              enable_glrlm=False,
              enable_glszm=False,
              enable_ngtdm=False,
              enable_gldm=False,
              binWidth=5.0)  # 一阶特征不依赖离散化，binWidth无关紧要

    # 如果第一步成功，再尝试启用纹理特征，并使用 binCount 固定灰度级数（例如 16）
    print("\n【第二步】启用纹理特征，使用 binCount=16 固定灰度级数")
    test_case(TEST_CASE,'CT',
              enable_firstorder=True,
              enable_shape=True,
              enable_glcm=True,
              enable_glrlm=True,
              enable_glszm=True,
              enable_ngtdm=True,
              enable_gldm=True,
              binCount=16)   # 固定灰度级数，避免内存爆炸

    # 如果第二步仍失败，可以尝试只启用 GLCM（最耗内存）并进一步减小 binCount
    # 或者增大 binWidth（减少灰度级数）