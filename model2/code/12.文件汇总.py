import os
import shutil
import re

def main():
    # 源根目录列表
    src_roots = [
        r'D:\gulianyu\LungAd_Radiomics\binarized_roi',
        r'D:\gulianyu\LungAd_Radiomics\converted_nifti',
        'D:\gulianyu\LungAd_Radiomics\\registered_nifti'
    ]
    # 目标根目录
    dst_root = r'D:\gulianyu\LungAd_Radiomics\total'

    # 创建目标根目录（如果不存在）
    os.makedirs(dst_root, exist_ok=True)

    total_copied = 0
    total_errors = 0

    for src_root in src_roots:
        src_dirname = os.path.basename(src_root)  # 仅用于提示，不再创建子目录
        if not os.path.isdir(src_root):
            print(f"警告：源目录不存在或不可读：{src_root}")
            continue

        print(f"\n处理源目录：{src_root}")
        try:
            items = os.listdir(src_root)
        except PermissionError:
            print(f"错误：无权限读取目录 {src_root}")
            total_errors += 1
            continue

        # 获取所有病例子文件夹（如0001、0002...）
        sub_dirs = [item for item in items if os.path.isdir(os.path.join(src_root, item))]
        if not sub_dirs:
            print(f"  未找到子文件夹")
            continue

        for sub in sub_dirs:
            src_sub = os.path.join(src_root, sub)
            dst_case = os.path.join(dst_root, sub)
            os.makedirs(dst_case, exist_ok=True)

            # 列出源病例文件夹下的所有直接文件（不包括子文件夹）
            try:
                files = [f for f in os.listdir(src_sub) if os.path.isfile(os.path.join(src_sub, f))]
            except PermissionError:
                print(f"  错误：无法读取目录 {src_sub}")
                total_errors += 1
                continue

            for file in files:
                #转换文件这里只复制CT，PET需要跳过
                if src_root == 'D:\gulianyu\LungAd_Radiomics\converted_nifti' and '_PET.nii.gz' in file:
                    continue
                src_file = os.path.join(src_sub, file)
                dst_file = os.path.join(dst_case, file)
                try:
                    shutil.copy2(src_file, dst_file)  # copy2保留元数据
                    print(f"    复制文件：{src_file} -> {dst_file}")
                    total_copied += 1
                except Exception as e:
                    print(f"    复制失败：{src_file} -> {dst_file}\n      错误：{e}")
                    total_errors += 1

    # print("开始复制roi")
    # src_dir = r'D:\gulianyu\LungAd_Radiomics\raw_roi'
    # copied = 0
    # errors = 0
    # # 遍历源目录中的所有文件
    # for filename in os.listdir(src_dir):
    #     filepath = os.path.join(src_dir, filename)
    #     if not os.path.isfile(filepath):
    #         continue  # 跳过子文件夹
    #
    #     # 解析数字前缀：格式应为 数字_ROI.nii.gz
    #     # 用正则提取开头的数字部分
    #     match = re.match(r'^(\d+)_ROI\.nii\.gz$', filename)
    #     if not match:
    #         print(f"跳过不匹配的文件: {filename}")
    #         continue
    #
    #     number = match.group(1)  # 数字字符串，如 '0001'
    #
    #     # 目标子文件夹
    #     dst_subdir = os.path.join(dst_root, number)
    #     os.makedirs(dst_subdir, exist_ok=True)
    #
    #     dst_file = os.path.join(dst_subdir, filename)
    #     try:
    #         shutil.copy2(filepath, dst_file)
    #         print(f"已复制: {filename} -> {dst_file}")
    #         copied += 1
    #     except Exception as e:
    #         print(f"复制失败: {filename} -> {dst_file}, 错误: {e}")
    #         errors += 1
    # print(f"\nroi 完成。成功复制 {copied} 个文件，失败 {errors} 个。")
    print("\n========== 汇总完成 ==========")
    print(f"成功复制文件数：{total_copied}")
    print(f"错误数：{total_errors}")

if __name__ == "__main__":
    main()