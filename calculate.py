import cv2
import numpy as np
import os


def calculate_white_pixel_ratio_in_mask(mask,img):
    """
    计算掩码（mask）图像内白色像素的占比（白色像素定义为255）
    注：mask应为二值图像（仅含0=黑色背景，255=白色目标区域）

    参数:
        mask_img: 输入的掩码图像（支持cv2读取的numpy数组、灰度图/单通道图）
                  可以是文件路径字符串，也可以是已加载的图像矩阵

    返回:
        white_ratio: 掩码内白色像素占比（0.0 ~ 1.0），若输入无效返回0.0
        white_count: 掩码内白色像素数量（int）
        total_valid_count: 掩码内有效像素总数（排除异常情况后的像素数，int）
    """
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 步骤3：提取「图像在掩码白色区域内」的像素（核心：保留掩码内的图像像素，掩码外置0）
    img_in_mask = cv2.bitwise_and(img, binary_mask)

    # 步骤4：统计掩码内的有效像素数（即掩码本身的白色像素数，这是分母）
    mask_valid_count = np.sum(binary_mask == 255)

    # 步骤5：统计「掩码内图像的白色像素数」（即img_in_mask中值为255的像素，这是分子）
    # 若需自定义图像白色阈值（如>200即为白色），可改为：img_white_count = np.sum(img_in_mask > 200)
    img_white_count = np.sum(img_in_mask >= 220)

    # 步骤4：计算占比（避免除以0错误）
    if mask_valid_count == 0:
        print("警告：掩码图像无有效像素")
        white_ratio = 0.0
    else:
        white_ratio = img_white_count / mask_valid_count
    return white_ratio, img_white_count, mask_valid_count


def traverse_folder(folder_path):
    """遍历文件夹处理所有图像"""
    os.makedirs("mask", exist_ok=True)
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(folder_path, filename)
        # 读取彩色图（兼容后续预处理）
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"无法读取图像 {filename}，跳过")
            continue
        mask_path = os.path.join("mask",filename)
        mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        try:
            white_ratio, img_white_count, mask_valid_count = calculate_white_pixel_ratio_in_mask(mask, img)
            print(white_ratio, img_white_count, mask_valid_count)
            print(f"{filename}处理完成")
        except Exception as e:
            print(f"处理 {filename} 出错：{e}")

if __name__ == "__main__":
    img_path = "result"  # 替换为你的图路径
    try:
        traverse_folder(img_path)
        print("区域填充mask已保存！")
    except Exception as e:
        print(f"出错：{e}")