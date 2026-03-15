# -*- coding: utf-8 -*-
import os
import SimpleITK as sitk
import numpy as np

from model2.code.contants import TEST_CASE_ID, TEST_SWITCH

# ===================== 配置全局路径 =====================
ROOT_DIR = r"D:\gulianyu\LungAd_Radiomics"
CT_CONVERTED_DIR = os.path.join(ROOT_DIR, "converted_nifti")
PET_SUV_DIR = os.path.join(ROOT_DIR, "pet_suv")
REGISTERED_DIR = os.path.join(ROOT_DIR, "registered_nifti")

def ensure_3d(image, image_name="图像"):
    """
    确保图像是3D单通道；如果是4D，提取第一个时间点；如果是向量图像，取第一通道。
    返回处理后的3D图像，如果无法处理则返回None。
    """
    dim = image.GetDimension()
    size = image.GetSize()
    pixel_type = image.GetPixelID()

    print(f"{image_name}: 维度={dim}, 尺寸={size}, 像素类型={image.GetPixelIDTypeAsString()}")

    # 如果是4D，尝试提取第一个时间点
    if dim == 4:
        # 假设4D形状为 (时间, z, y, x) 或 (z, y, x, 时间)？SimpleITK中顺序通常是 (x,y,z,t) 或 (x,y,z,components)
        # 这里通过数组转换处理
        arr = sitk.GetArrayFromImage(image)  # 返回 (t, z, y, x) 顺序
        if len(arr.shape) == 4:
            # 取第一个时间点
            arr_3d = arr[0, ...]  # shape (z, y, x)
            new_img = sitk.GetImageFromArray(arr_3d)
            new_img.SetSpacing(image.GetSpacing()[:3])  # 只取前三个间距
            new_img.SetOrigin(image.GetOrigin()[:3])
            new_img.SetDirection(image.GetDirection()[:9])  # 3x3方向矩阵
            print(f"✅ 已将4D图像转换为3D，新尺寸={new_img.GetSize()}")
            return new_img
        else:
            print(f"❌ 无法处理4D图像，数组形状={arr.shape}")
            return None

    # 如果是向量图像（多通道），取第一个通道
    elif dim == 3 and image.GetNumberOfComponentsPerPixel() > 1:
        print(f"{image_name} 是多通道图像，取第一个通道")
        # 使用VectorIndexSelectionCastImageFilter提取第一个通道
        selector = sitk.VectorIndexSelectionCastImageFilter()
        selector.SetIndex(0)
        return selector.Execute(image)

    # 已经是3D单通道
    elif dim == 3 and image.GetNumberOfComponentsPerPixel() == 1:
        return image

    else:
        print(f"❌ 不支持的图像维度: {dim}")
        return None


def resample_pet_to_ct(case_id):
    """
    修改为：直接基于硬件物理坐标系将PET重采样到CT网格
    """
    ct_path = os.path.join(CT_CONVERTED_DIR, case_id, f"{case_id}_CT.nii.gz")
    pet_path = os.path.join(PET_SUV_DIR, case_id, f"{case_id}_PET_SUV.nii.gz")
    output_pet_path = os.path.join(REGISTERED_DIR, case_id, f"{case_id}_PET_registered.nii.gz")

    if not os.path.exists(ct_path) or not os.path.exists(pet_path):
        print(f"❌ 病例{case_id}：文件缺失，请检查")
        return False

    os.makedirs(os.path.join(REGISTERED_DIR, case_id), exist_ok=True)

    try:
        # 1. 读取图像
        ct_image_raw = sitk.ReadImage(ct_path)
        pet_image_raw = sitk.ReadImage(pet_path)

        # 2. 确保3D (直接用你原本的函数)
        ct_image = ensure_3d(ct_image_raw, "CT")
        pet_image = ensure_3d(pet_image_raw, "PET")
        if ct_image is None or pet_image is None:
            return False

        # 3. 设置重采样器 (Resampler)
        resampler = sitk.ResampleImageFilter()

        # 【核心步骤】：以CT作为参考图像 (Reference Image)
        # 这会自动提取CT的 Size, Spacing, Origin 和 Direction
        resampler.SetReferenceImage(ct_image)

        # 设置插值方法
        # 对于连续数值(SUV)，推荐使用 B-Spline 或 线性插值。
        # B-Spline (sitkBSpline) 在处理医学图像数值时更平滑，线性(sitkLinear)也可以
        resampler.SetInterpolator(sitk.sitkBSpline)

        # 默认的 Transform 就是 IdentityTransform (也就是不发生额外的位移)
        resampler.SetTransform(sitk.Transform())

        # 如果超出边界的像素，填充为 0
        resampler.SetDefaultPixelValue(0)

        # 4. 执行重采样
        print(f"⏳ 病例{case_id}：正在将 PET 重采样到 CT 网格...")
        resampled_pet = resampler.Execute(pet_image)

        # 5. 保存结果
        sitk.WriteImage(resampled_pet, output_pet_path)
        print(f"✅ 病例{case_id}：重采样完成 -> {output_pet_path}")
        return True

    except Exception as e:
        print(f"❌ 病例{case_id}：重采样失败 -> {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def batch_register_all_cases():
    case_folders = [f for f in os.listdir(CT_CONVERTED_DIR) if os.path.isdir(os.path.join(CT_CONVERTED_DIR, f))]
    case_folders.sort()

    if not case_folders:
        print(f"⚠️ 在 {CT_CONVERTED_DIR} 中没有找到病例文件夹")
        return

    print(f"📌 开始批量配准，共 {len(case_folders)} 个病例\n")
    success, fail = 0, 0
    for case_id in case_folders:
        case_id_int = int(case_id)
        if TEST_SWITCH and case_id_int not in TEST_CASE_ID:
            continue
        print(f"正在处理: {case_id}")
        if resample_pet_to_ct(case_id):
            success += 1
        else:
            fail += 1
        print("-" * 50)

    print(f"\n🎉 批量配准完成！成功={success}, 失败={fail}")

if __name__ == "__main__":
    batch_register_all_cases()