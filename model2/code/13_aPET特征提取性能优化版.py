# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import multiprocessing
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ===================== 配置全局路径 =====================
ROOT_DIR = r"D:\gulianyu\LungAd_Radiomics"
PROCESSED_DIR = os.path.join(ROOT_DIR, "compilation_data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "features_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 优化配置
CPU_CORES = multiprocessing.cpu_count() - 2
RESAMPLE_SPACING = [2, 2, 2]
MAX_WORKERS = min(8, CPU_CORES)


# ===================== 1. 修复提取器（核心解决特征缺失） =====================
def get_optimized_extractor():
    """
    正确配置提取器：
    1. 按官方规范启用特征类
    2. 适配PET数据值域
    3. 保留所有核心特征类
    """
    # 第一步：初始化提取器（仅传基础全局参数）
    extractor = featureextractor.RadiomicsFeatureExtractor(
        enableCExtensions=True,  # 启用C++加速
        threads=CPU_CORES  # 多线程计算
    )

    # 第二步：全局设置（适配PET数据）
    # 1. 重采样（减少计算量，不影响特征完整性）
    extractor.settings['resampledPixelSpacing'] = RESAMPLE_SPACING
    extractor.settings['interpolator'] = sitk.sitkLinear
    extractor.settings['resampleInterpolator'] = sitk.sitkLinear

    # 2. PET数据专用设置（关键：解决特征计算被跳过的问题）
    extractor.settings['normalize'] = True  # PET值需归一化（之前关闭导致特征计算失败）
    extractor.settings['normalizeScale'] = 1000  # 适配PET SUV值范围（0-10）
    extractor.settings['binWidth'] = 25  # 增加分箱数（PET值范围大，默认16不够）
    extractor.settings['binCount'] = 32  # 强制分箱数量，确保纹理特征能计算

    # 3. ROI设置（保留）
    extractor.settings['maskedKernel'] = True
    extractor.settings['minimumROISize'] = 10  # 降低最小ROI阈值（避免小病灶被过滤）
    extractor.settings['maskPixelValue'] = 1

    # 第三步：特征类配置（按官方规范手动启用，核心修复！）
    extractor.disableAllFeatures()  # 先禁用所有

    # 启用所有核心特征类（肺癌PET-CT必用）
    extractor.enableFeatureClassByName('firstorder')  # 一阶统计特征（均值、方差、熵等）
    extractor.enableFeatureClassByName('shape')  # 形状特征（保留，你原本能提取的）
    extractor.enableFeatureClassByName('glcm')  # 灰度共生矩阵（核心纹理）
    extractor.enableFeatureClassByName('glrlm')  # 灰度游程矩阵
    extractor.enableFeatureClassByName('glszm')  # 灰度大小区域矩阵
    extractor.enableFeatureClassByName('ngtdm')  # 邻域灰度差矩阵
    extractor.enableFeatureClassByName('gldm')  # 灰度依赖矩阵（可选，增加纹理维度）

    # 第四步：GLCM参数优化（不影响特征数量，仅提速）
    try:
        extractor.settings['glcm']['distance'] = [1, 2]  # 保留2个距离（平衡速度和特征数）
        extractor.settings['glcm']['angles'] = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 全角度
    except:
        pass  # 兼容不同版本

    return extractor


# ===================== 2. 单病例提取（保留完整特征） =====================
def extract_single_case_optimized(args):
    """单病例提取：确保特征完整提取"""
    case_id, modality = args

    # 每个进程独立初始化提取器
    try:
        extractor = get_optimized_extractor()
    except Exception as e:
        print(f"❌ {case_id}：提取器初始化失败 → {str(e)[:50]}")
        return None

    # 路径构建
    img_path = os.path.join(PROCESSED_DIR, case_id, f"{case_id}_{modality}.nii.gz")
    if modality == 'PET':
        img_path = os.path.join(PROCESSED_DIR, case_id, f"{case_id}_{modality}_registered.nii.gz")
    roi_path = os.path.join(PROCESSED_DIR, case_id, f"{case_id}_ROI_bin.nii.gz")

    # 文件检查
    if not os.path.exists(img_path):
        print(f"❌ {case_id}：{modality}文件缺失 → {img_path}")
        return None
    if not os.path.exists(roi_path):
        print(f"❌ {case_id}：ROI文件缺失 → {roi_path}")
        return None

    # 核心提取逻辑（增加调试信息）
    try:
        # 执行提取（返回完整特征字典）
        features = extractor.execute(img_path, roi_path)

        # 过滤特征：保留所有非诊断类特征（包括一阶、纹理、形状）
        feature_dict = {'case_id': case_id}
        feature_count = 0
        for k, v in features.items():
            # 排除诊断信息，仅保留数值型特征
            if not k.startswith('diagnostics_') and isinstance(v, (int, float)):
                feature_dict[k] = v
                feature_count += 1

        print(f"✅ {case_id}：提取完成，共{feature_count}个特征")
        return feature_dict
    except Exception as e:
        print(f"❌ {case_id}：提取失败 → {str(e)[:100]}")
        return None


# ===================== 3. 批量提取（保留多进程，确保特征完整） =====================
def batch_extract_features_optimized():
    # 获取病例列表
    case_folders = [
        f for f in os.listdir(PROCESSED_DIR)
        if os.path.isdir(os.path.join(PROCESSED_DIR, f))
    ]
    case_folders.sort()
    print(f"📌 开始提取，共{len(case_folders)}个病例，进程数：{MAX_WORKERS}\n")

    # 构建任务
    pet_tasks = [(case_id, 'PET') for case_id in case_folders]
    pet_features_list = []

    # 多进程提取（保留降级方案）
    try:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_case = {executor.submit(extract_single_case_optimized, t): t[0] for t in pet_tasks}
            for future in as_completed(future_to_case):
                res = future.result()
                if res:
                    pet_features_list.append(res)
    except Exception as e:
        print(f"\n⚠️  多进程出错 → {str(e)}，切换到单进程模式\n")
        # 单进程降级
        for task in pet_tasks:
            res = extract_single_case_optimized(task)
            if res:
                pet_features_list.append(res)

    # 保存结果（显示特征数量）
    if pet_features_list:
        pet_df = pd.DataFrame(pet_features_list)
        output_path = os.path.join(OUTPUT_DIR, "pet_features_full.csv")
        pet_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n✅ 提取完成！保存路径：{output_path}")
        print(f"✅ 有效病例数：{len(pet_features_list)}")
        print(f"✅ 提取特征总数：{pet_df.shape[1] - 1}（不含case_id列）")  # 显示特征列数
    else:
        print("\n❌ 无有效提取结果！")

    print("\n🎉 提取流程结束")


# ===================== 应急单进程版本（确保特征完整） =====================
def batch_extract_single_process():
    """纯单进程版本（最稳定，确保特征完整）"""
    extractor = get_optimized_extractor()
    case_folders = [f for f in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, f))]
    case_folders.sort()
    print(f"📌 单进程提取开始，共{len(case_folders)}个病例\n")

    pet_features_list = []
    for case_id in case_folders:
        img_path = os.path.join(PROCESSED_DIR, case_id, f"{case_id}_PET_registered.nii.gz")
        roi_path = os.path.join(PROCESSED_DIR, case_id, f"{case_id}_ROI_bin.nii.gz")

        if not (os.path.exists(img_path) and os.path.exists(roi_path)):
            print(f"❌ {case_id}：文件缺失")
            continue

        try:
            features = extractor.execute(img_path, roi_path)
            feature_dict = {'case_id': case_id}
            feature_count = 0
            for k, v in features.items():
                if not k.startswith('diagnostics_') and isinstance(v, (int, float)):
                    feature_dict[k] = v
                    feature_count += 1
            pet_features_list.append(feature_dict)
            print(f"✅ {case_id}：完成，特征数：{feature_count}")
        except Exception as e:
            print(f"❌ {case_id}：失败 → {str(e)[:50]}")

    # 保存
    if pet_features_list:
        pet_df = pd.DataFrame(pet_features_list)
        pet_df.to_csv(
            os.path.join(OUTPUT_DIR, "pet_features_single_full.csv"),
            index=False, encoding='utf-8'
        )
        print(f"\n✅ 单进程提取完成")
        print(f"✅ 有效病例数：{len(pet_features_list)}")
        print(f"✅ 提取特征总数：{pet_df.shape[1] - 1}")
    else:
        print("\n❌ 无有效结果")


# ===================== 运行主函数 =====================
if __name__ == "__main__":
    os.environ['RADIOMICS_ENABLE_C_EXTENSIONS'] = '1'

    # 优先运行多进程版本
    batch_extract_features_optimized()

    # 如果仍有问题，启用单进程版本
    # batch_extract_single_process()