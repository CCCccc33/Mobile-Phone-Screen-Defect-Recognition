import cv2
import numpy as np


def calculate_average_gray(image_path):
    """
    计算图片的平均灰度值
    参数:
        image_path: 图片文件路径
    返回:
        float: 图片的平均灰度值（范围 0-255，0=纯黑，255=纯白）
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"读取本地图片失败，请检查路径是否正确：{image_path}")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_gray = np.mean(gray_img)

    return round(avg_gray, 2)  # 保留两位小数，方便查看


# ------------------- 测试示例 -------------------
if __name__ == "__main__":
    img1_url = "Oil_0003.jpg"
    img2_url = "Oil_0011.jpg"

    # 计算两张图的平均灰度值
    try:
        avg_gray1 = calculate_average_gray(img1_url)
        avg_gray2 = calculate_average_gray(img2_url)
        print(f"第一张图（全黑手机）平均灰度值：{avg_gray1}")
        print(f"第二张图（带白色部分手机）平均灰度值：{avg_gray2}")
        print(f"灰度值差值：{round(abs(avg_gray1 - avg_gray2), 2)}")

        # 简单的区分逻辑（可根据实际测试结果调整阈值）
        threshold = 65  # 阈值可根据你的样本调整
        if avg_gray1 < threshold:
            print("判断：第一张图为全黑手机")
        else:
            print("判断：第一张图为带白色部分的手机")

        if avg_gray2 < threshold:
            print("判断：第二张图为全黑手机")
        else:
            print("判断：第二张图为带白色部分的手机")
    except Exception as e:
        print(f"计算失败：{e}")