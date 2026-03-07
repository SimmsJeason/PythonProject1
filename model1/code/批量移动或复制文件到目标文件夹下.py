import os
import shutil
import sys

def collect_files(root_dir, dest_subdir="raw_roi", source_subdir="", action="copy"):
    """
    收集根目录下每个子文件夹中的 <子文件夹名>.nii 文件到目标文件夹。
    """
    dest_dir = os.path.join(root_dir, dest_subdir)
    os.makedirs(dest_dir, exist_ok=True)

    print(f"目标文件夹: {dest_dir}")
    print(f"源文件位置: 每个子文件夹下的 {source_subdir + '/' if source_subdir else ''}<子文件夹名>.nii")
    print(f"操作: {'复制' if action == 'copy' else '移动'}\n")

    for item in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, item)
        if not os.path.isdir(subfolder_path):
            continue

        # 根据 source_subdir 构建完整源文件路径
        if source_subdir:
            src_file = os.path.join(subfolder_path, source_subdir, item + "_ROI.nii.gz")
        else:
            src_file = os.path.join(subfolder_path, item + "_ROI.nii.gz")

        if not os.path.isfile(src_file):
            print(f"跳过 {item}：未找到文件 {item}_ROI.nii.gz" + (f" 在 {source_subdir} 内" if source_subdir else ""))
            continue

        dst_file = os.path.join(dest_dir, item + "_ROI.nii.gz")
        if os.path.exists(dst_file):
            print(f"跳过 {item}：目标文件已存在 - {dst_file}")
            continue

        try:
            if action == "copy":
                shutil.copy2(src_file, dst_file)
                print(f"已复制: {item}_ROI.nii.gz")
            else:
                shutil.move(src_file, dst_file)
                print(f"已移动: {item}_ROI.nii.gz")
        except Exception as e:
            print(f"处理 {item} 时出错: {e}")

    print("\n收集完成！")

def main():
    # ===== 请修改为你的根目录路径 =====
    root_dir = r"D:\gulianyu\LungAd_Radiomics\raw_data"  # 例如 D:\LungAd_Radiomics
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]

    if not os.path.isdir(root_dir):
        print(f"错误：根目录不存在 - {root_dir}")
        return

    # ===== 配置选项 =====
    action = "copy"           # 可选 "copy"（复制）或 "move"（移动）
    source_subdir = "ROI"      # 因为文件在 001/ROI/001.nii，所以填 "ROI"
    dest_subdir = "D:\gulianyu\LungAd_Radiomics\\raw_roi"    # 目标文件夹名，将在根目录下创建

    collect_files(root_dir, dest_subdir, source_subdir, action)

if __name__ == "__main__":
    main()
    error_ID=[1027,1056,1061,1093,]