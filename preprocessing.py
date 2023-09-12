import os
import cv2
import numpy as np

# 指定存放图片的主目录
main_directory = '/home2/mty/project/datasets/ID_UPDATE'

# 指定保存处理后图片的主目录
output_main_directory = '/home2/mty/project/datasets/rotation'

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
            # original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
            
            # 在黑色画布上粘贴原始图片
            height, width = 1000, 1000
            black_canvas = np.zeros((height,width,3), dtype=np.uint8)
            rows, cols = original_img.shape[:2]
            start_x, start_y = width//2 - cols//2, height//2 - rows//2
            black_canvas[start_y:start_y+rows, start_x:start_x+cols] = original_img



            gray_scale_img = cv2.cvtColor(black_canvas,cv2.COLOR_RGB2GRAY)
            _, threshold_img = cv2.threshold(gray_scale_img, 0, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            rectangle = cv2.minAreaRect(largest_contour)
            center, size, angle = rectangle
            angle_points = cv2.boxPoints(rectangle).astype(int)

            # The cropped and rotated image
            if size[0] < size[1]:
                rotate_angle = angle + 90
                crop_h = int(size[1])
                crop_w = int(size[0])
            else:
                rotate_angle = angle
                crop_h = int(size[0])
                crop_w = int(size[1])

            rotated_img = cv2.warpAffine(black_canvas, cv2.getRotationMatrix2D(center, rotate_angle, 1.0), (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            cropped_img = cv2.getRectSubPix(rotated_img, (crop_h, crop_w), center)

            left_img = cv2.getRectSubPix(cropped_img, (crop_h//2, crop_w), (crop_h//4, crop_w//2))
            right_img = cv2.getRectSubPix(cropped_img, (crop_h//2, crop_w), (crop_h*3//4, crop_w//2))
            l_count, r_count = 0, 0
            for l_1 in left_img:
                for l_2 in l_1:
                    if l_2.sum() != 0:
                        l_count+=1

            for r_1 in right_img:
                for r_2 in r_1:
                    if r_2.sum() != 0:
                        r_count+=1

            if l_count > r_count:
                aligned_img = cv2.warpAffine(cropped_img, cv2.getRotationMatrix2D((crop_h//2, crop_w//2), 180, 1.0), (crop_h, crop_w), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            else:
                aligned_img = cropped_img

            # 提取蓝色通道并二值化
            cyan_scale_image = np.zeros_like(aligned_img)
            cyan_scale_image[:, :, 0] = aligned_img[:, :, 0]
            cyan_scale_image[:, :, 1] = aligned_img[:, :, 1]
            _, binary_img = cv2.threshold(cyan_scale_image, 190, 255, cv2.THRESH_BINARY)

            # 构建输出图片路径，使用原始图片的文件名
            output_image_path = os.path.join(output_subdirectory, image_file)
            # cv2.imwrite(output_image_path, binary_img)
            cv2.imwrite(output_image_path, aligned_img)

print("图片处理完成并已保存到指定目录。")