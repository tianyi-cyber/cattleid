import os
import cv2
import numpy as np

# 指定存放图片的主目录
main_directory = '/home2/mty/project/datasets/rotation-selected'

# 指定保存处理后图片的主目录
output_main_directory = '/home2/mty/project/datasets/cyan-selected'

# 创建保存主目录（如果不存在）
os.makedirs(output_main_directory, exist_ok=True)

# 遍历主目录下的所有子文件夹
for i, folder_name in enumerate(os.listdir(main_directory)):
    # 构建子文件夹的完整路径
    folder_path = os.path.join(main_directory, folder_name)

    # 检查路径是否是一个目录，并且不是隐藏文件夹
    if os.path.isdir(folder_path) and not folder_name.startswith('.'):
        # 创建保存子文件夹中处理后图片的目录，直接使用输入文件夹的文件名
        output_subdirectory = os.path.join(output_main_directory, folder_name)
        os.makedirs(output_subdirectory, exist_ok=True)

        # 获取图片文件路径列表
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

        # 循环处理每张图片
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)

            # 读取原始图片
            original_img = cv2.imread(image_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB

            # 提取蓝色通道并二值化
            cyan_scale_image = np.zeros_like(original_img)
            cyan_scale_image[:, :, 0] = original_img[:, :, 0]
            cyan_scale_image[:, :, 1] = original_img[:, :, 1]
            _, binary_img = cv2.threshold(cyan_scale_image, 190, 255, cv2.THRESH_BINARY)

            # 构建输出图片路径，使用原始图片的文件名
            output_image_path = os.path.join(output_subdirectory, image_file)
            cv2.imwrite(output_image_path, binary_img)
          

print("图片处理完成并已保存到指定目录。")