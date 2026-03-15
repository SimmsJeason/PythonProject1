import os
import sys
import logging
import traceback
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from model2.code.contants import TEST_CASE_ID, TEST_SWITCH

# ===================== 全局配置与路径 =====================
ROOT_DIR = r"D:\gulianyu\LungAd_Radiomics"
PROCESSED_DIR = os.path.join(ROOT_DIR, "compilation_data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "features_output")
LOG_DIR = os.path.join(ROOT_DIR, "log")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

  # ，。
# ；；P===================== 日志设置 =====================
def setup_logger():
    logger = logging.getLogger('RadiomicsBatch')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'radiomics_extraction.log'), encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logger()


# ===================== 特征提取器工厂 =====================
def get_extractor(enable_firstorder=True, enable_shape=True,
                  enable_glcm=True, enable_glrlm=True,
                  enable_glszm=True, enable_ngtdm=True,
                  enable_gldm=True,
                  binWidth=None, binCount=16):
    extractor = featureextractor.RadiomicsFeatureExtractor(
        enableCExtensions=True,
        threads=1
    )

    extractor.settings['resampledPixelSpacing'] = [2, 2, 2]
    extractor.settings['interpolator'] = sitk.sitkLinear
    extractor.settings['resampleInterpolator'] = sitk.sitkLinear

    if binCount is not None:
        extractor.settings['binCount'] = binCount
    elif binWidth is not None:
        extractor.settings['binWidth'] = binWidth

    extractor.settings['maskedKernel'] = True
    extractor.settings['minimumROISize'] = 10
    extractor.settings['maskPixelValue'] = 1

    extractor.disableAllFeatures()
    if enable_firstorder: extractor.enableFeatureClassByName('firstorder')
    if enable_shape: extractor.enableFeatureClassByName('shape')
    if enable_glcm: extractor.enableFeatureClassByName('glcm')
    if enable_glrlm: extractor.enableFeatureClassByName('glrlm')
    if enable_glszm: extractor.enableFeatureClassByName('glszm')
    if enable_ngtdm: extractor.enableFeatureClassByName('ngtdm')
    if enable_gldm: extractor.enableFeatureClassByName('gldm')

    return extractor


# ===================== 单病例处理逻辑 (修改版：不再保存单独CSV) =====================
def process_single_case(case_id, modality, extractor_params):
    log_prefix = f"[Case {case_id} {modality}]"

    try:
        # 1. 构建路径
        if modality == 'PET':
            img_filename = f"{case_id}_PET_registered.nii.gz"
        else:
            img_filename = f"{case_id}_CT.nii.gz"

        img_path = os.path.join(PROCESSED_DIR, case_id, img_filename)
        roi_path = os.path.join(PROCESSED_DIR, case_id, f"{case_id}_ROI_bin.nii.gz")

        if not os.path.exists(img_path) or not os.path.exists(roi_path):
            logger.error(f"{log_prefix} 文件缺失")
            return None

        # 2. 初始化提取器
        extractor = get_extractor(**extractor_params)

        # 3. 执行提取
        logger.info(f"{log_prefix} 开始提取...")
        features = extractor.execute(img_path, roi_path)

        # 4. 整理结果
        feat_dict = {'case_id': case_id}
        feature_count = 0
        for k, v in features.items():
            if not k.startswith('diagnostics_'):
                feat_dict[k] = v
                feature_count += 1

        logger.info(f"{log_prefix} 成功提取 {feature_count} 个特征")
        return feat_dict

    except MemoryError:
        logger.error(f"{log_prefix} 内存溢出 (OOM)")
        return None
    except Exception as e:
        logger.error(f"{log_prefix} 发生错误: {str(e)}")
        logger.debug(traceback.format_exc())
        return None


# ===================== 批量调度主函数 (修改版：按模态分开汇总) =====================
def main():
    logger.info("=" * 30)
    logger.info("开始批量影像组学特征提取")
    logger.info(f"数据目录: {PROCESSED_DIR}")

    # 1. 准备任务列表
    if not os.path.exists(PROCESSED_DIR):
        logger.error(f"数据目录不存在: {PROCESSED_DIR}")
        return

    all_cases = [d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))]
    #all_cases = all_cases[:3] #测试用
    if not all_cases:
        logger.warning("未找到任何病例数据文件夹！")
        return

    modalities_to_process = ['CT', 'PET']
    tasks = []
    for case in all_cases:
        case_id_int = int(case)
        if TEST_SWITCH and case_id_int not in TEST_CASE_ID:
            continue
        for mod in modalities_to_process:
            tasks.append((case, mod))

    logger.info(f"共发现 {len(all_cases)} 个病例，计划处理 {len(tasks)} 个任务")

    # 2. 配置提取参数
    extractor_config = {
        'enable_firstorder': True,
        'enable_shape': True,
        'enable_glcm': True,
        'enable_glrlm': True,
        'enable_glszm': True,
        'enable_ngtdm': True,
        'enable_gldm': True,
        'binCount': 16,
    }

    # 3. 多线程执行
    all_results = []
    max_workers = 8

    logger.info(f"启动线程池，最大工作线程数: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(process_single_case, case, mod, extractor_config): (case, mod)
            for case, mod in tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="总体进度"):
            case_id, modality = future_to_task[future]
            res = future.result()
            if res is not None:
                # 在结果中标记模态，用于后续分组
                res['modality'] = modality
                all_results.append(res)

    # 4. 最终汇总与分开保存 (核心修改点)
    if all_results:
        final_df = pd.DataFrame(all_results)

        # 分别筛选 CT 和 PET
        ct_df = final_df[final_df['modality'] == 'CT'].copy()
        pet_df = final_df[final_df['modality'] == 'PET'].copy()

        # 保存 CT 特征 (删除 modality 列)
        if not ct_df.empty:
            ct_df.drop(columns=['modality'], inplace=True)
            ct_output_path = os.path.join(OUTPUT_DIR, "ct_features.csv")
            ct_df.to_csv(ct_output_path, index=False)
            logger.info(f"CT 特征汇总完成，共 {len(ct_df)} 个病例，已保存至: {ct_output_path}")

        # 保存 PET 特征 (删除 modality 列)
        if not pet_df.empty:
            pet_df.drop(columns=['modality'], inplace=True)
            pet_output_path = os.path.join(OUTPUT_DIR, "pet_features.csv")
            pet_df.to_csv(pet_output_path, index=False)
            logger.info(f"PET 特征汇总完成，共 {len(pet_df)} 个病例，已保存至: {pet_output_path}")

        logger.info("=" * 30)
        logger.info(f"全部流程结束。")
    else:
        logger.warning("没有成功提取任何特征，请检查日志。")


if __name__ == "__main__":
    main()