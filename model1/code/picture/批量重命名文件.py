import os
import sys

def main():
    # ===== 配置选项 =====
    # 选择目标文件存放位置：
    #   "parent" - 放在子文件夹（如001）下
    #   "roi"    - 仍放在ROI文件夹内
    TARGET_LOCATION = "roi"  # 或 "roi"

    # 方法1：直接在代码中指定根目录（请修改为你的实际路径）
    root_dir = r"D:\gulianyu\LungAd_Radiomics\raw_data"  # 替换为你的根目录路径

    # 方法2：通过命令行参数获取根目录（如果提供了参数）
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]

    # 检查根目录是否存在
    if not os.path.isdir(root_dir):
        print(f"错误：根目录不存在或不是一个文件夹 - {root_dir}")
        return

    print(f"开始处理根目录: {root_dir}")
    print(f"目标文件位置: {'子文件夹下' if TARGET_LOCATION == 'parent' else 'ROI文件夹内'}\n")

    # 遍历根目录下的所有项目
    for item in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, item)

        # 只处理子文件夹（如001、002...）
        if not os.path.isdir(subfolder_path):
            continue

        # 构建 ROI 文件夹路径和源文件路径
        roi_path = os.path.join(subfolder_path, "ROI")
        src_file = os.path.join(roi_path, "CT.uint16.nii.gz")

        # 检查 ROI 文件夹和 CT.uint16.nii.gz 是否存在
        if not os.path.isdir(roi_path):
            print(f"跳过 {item}：未找到 ROI 文件夹")
            continue
        if not os.path.isfile(src_file):
            print(f"跳过 {item}：ROI 文件夹中未找到 CT.uint16.nii.gz")
            continue

        # 生成新文件名：子文件夹名 + .nii.gz
        new_filename = item + "_ROI.nii.gz"

        # 根据配置确定目标路径
        if TARGET_LOCATION == "parent":
            # 目标文件放在子文件夹下（与ROI同级）
            dst_file = os.path.join(subfolder_path, new_filename)
        else:
            # 目标文件仍放在 ROI 文件夹内
            dst_file = os.path.join(roi_path, new_filename)

        # 如果目标文件已存在，给出警告（os.rename 会直接覆盖）
        if os.path.exists(dst_file):
            print(f"警告：目标文件已存在，将覆盖 - {dst_file}")

        try:
            # 执行重命名（或移动+重命名）
            os.rename(src_file, dst_file)
            print(f"成功：{item}/ROI/CT.uint16.nii.gz -> {os.path.basename(dst_file)}")
        except Exception as e:
            print(f"失败：{item} - {e}")

    print("\n所有处理完成！")

if __name__ == "__main__":
    main()