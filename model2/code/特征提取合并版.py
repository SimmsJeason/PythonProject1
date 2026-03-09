# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置路径 =====================
ROOT_DIR = r"D:\gulianyu\LungAd_Radiomics"
PROCESSED_DIR = os.path.join(ROOT_DIR, "compilation_data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "features_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_WORKERS = 8  # 并行线程数

# ===================== 提取器工厂（根据模态配置）=====================
def get_extractor(modality='CT'):
    """
    根据模态返回配置好的提取器
    - CT: 使用 binWidth=25（标准CT值范围大）
    - PET: 使用 binCount=16（固定灰度级，避免内存爆炸）
    """
    extractor = featureextractor.RadiomicsFeatureExtractor(
        enableCExtensions=True,
        threads=1  # 每个进程单线程，因为使用进程池并行
    )

    # 重采样设置（所有模态一致）
    extractor.settings['resampledPixelSpacing'] = [2, 2, 2]
    extractor.settings['interpolator'] = sitk.sitkLinear
    extractor.settings['resampleInterpolator'] = sitk.sitkLinear

    # 关闭归一化（假设图像已经是SUV或CT值，无需再归一化）
    extractor.settings['normalize'] = False

    # 其他通用设置
    extractor.settings['maskedKernel'] = True
    extractor.settings['minimumROISize'] = 10
    extractor.settings['maskPixelValue'] = 1

    # 根据模态设置离散化参数
    if modality.upper() == 'CT':
        # CT 值范围大，使用固定 bin width
        extractor.settings['binWidth'] = 25.0
    elif modality.upper() == 'PET':
        # PET（SUV）范围较小，使用固定灰度级数以控制内存
        extractor.settings['binCount'] = 16
    else:
        raise ValueError(f"未知模态: {modality}")

    # 启用所有特征类
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('shape')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('ngtdm')
    extractor.enableFeatureClassByName('gldm')

    # 可选：限制GLCM计算量（减少内存占用）
    try:
        extractor.settings['glcm'] = {'distance': [1]}
        extractor.settings['glcm']['angles'] = [0, np.pi/2]
    except:
        pass

    return extractor

# ===================== 单病例单模态提取 =====================
def extract_single_case(args):
    """参数: (case_id, modality)"""
    case_id, modality = args

    # 构建文件路径
    if modality.upper() == 'CT':
        img_path = os.path.join(PROCESSED_DIR, case_id, f"{case_id}_CT.nii.gz")
    else:  # PET
        img_path = os.path.join(PROCESSED_DIR, case_id, f"{case_id}_PET_registered.nii.gz")
    roi_path = os.path.join(PROCESSED_DIR, case_id, f"{case_id}_ROI_bin.nii.gz")

    # 检查文件是否存在
    if not os.path.exists(img_path):
        print(f"❌ {case_id} {modality} 图像缺失: {img_path}")
        return None
    if not os.path.exists(roi_path):
        print(f"❌ {case_id} ROI 缺失: {roi_path}")
        return None

    # 每个进程独立创建提取器
    try:
        extractor = get_extractor(modality)
    except Exception as e:
        print(f"❌ {case_id} {modality} 提取器初始化失败: {e}")
        return None

    # 执行提取
    try:
        features = extractor.execute(img_path, roi_path)
    except MemoryError:
        print(f"❌ {case_id} {modality} 内存不足，跳过")
        return None
    except Exception as e:
        print(f"❌ {case_id} {modality} 提取失败: {e}")
        traceback.print_exc()
        return None

    # 构建结果字典（过滤诊断信息）
    feat_dict = {'case_id': case_id}
    for k, v in features.items():
        if not k.startswith('diagnostics_'):
            feat_dict[k] = v

    print(f"✅ {case_id} {modality} 提取完成，特征数: {len(feat_dict)-1}")
    return feat_dict

# ===================== 批量提取（并行）=====================
def batch_extract(modality, case_list):
    """提取指定模态的所有病例特征"""
    print(f"\n===== 开始批量提取 {modality} 特征，共 {len(case_list)} 个病例 =====")

    tasks = [(case_id, modality) for case_id in case_list]
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_case = {executor.submit(extract_single_case, task): task[0] for task in tasks}
        for future in as_completed(future_to_case):
            case_id = future_to_case[future]
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                print(f"❌ {case_id} {modality} 处理时发生异常: {e}")

    # 保存结果
    if results:
        df = pd.DataFrame(results)
        output_file = os.path.join(OUTPUT_DIR, f"{modality.lower()}_features.csv")
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ {modality} 特征已保存至: {output_file}")
        print(f"   有效病例数: {len(results)}，特征总数: {df.shape[1]-1}")
    else:
        print(f"❌ {modality} 无有效提取结果")

    return results

# ===================== 主流程 =====================
if __name__ == "__main__":
    # 获取所有病例文件夹
    case_folders = [
        f for f in os.listdir(PROCESSED_DIR)
        if os.path.isdir(os.path.join(PROCESSED_DIR, f))
    ]
    case_folders.sort()
    print(f"📌 发现 {len(case_folders)} 个病例文件夹")

    # 分别提取CT和PET特征
    batch_extract('CT', case_folders)
    batch_extract('PET', case_folders)

    print("\n🎉 全部提取完成！")