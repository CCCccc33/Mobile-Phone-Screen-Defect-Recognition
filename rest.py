import cv2
import numpy as np

def analyze_largest_white_contour(binary_mask):
    """
    从mask图像中提取最大白色轮廓，并计算其长宽比、轮廓面积、包围矩形面积

    参数:
        mask_path: mask图像文件路径（二值图像，白色为前景255，黑色为背景0）

    返回:
        max_contour: 最大白色轮廓
        aspect_ratio: 长宽比（宽/高）
        contour_area: 轮廓实际面积
        rect_area: 包围轮廓的最小矩形面积
    """
    contours, hierarchy = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("mask图像中未检测到任何白色轮廓！")

    max_contour = max(contours, key=cv2.contourArea)
    rotated_rect = cv2.minAreaRect(max_contour)  # 获取旋转矩形参数
    box_vertices = cv2.boxPoints(rotated_rect)  # 转换为4个顶点
    box_vertices = np.int0(box_vertices)  # 转为整数像素坐标

    # 4. 提取旋转矩形的关键信息（长宽、面积、旋转角度）
    (cx, cy), (w, h), angle = rotated_rect
    rotated_rect_area = w * h  # 旋转矩形面积（最小包围矩形面积）
    contour_area = cv2.contourArea(max_contour)  # 轮廓实际面积
    # 旋转矩形的长宽比（统一按 长/宽 计算，确保大于等于1）
    aspect_ratio_rotated = max(w, h) / min(w, h) if min(w, h) != 0 else 1.0
    return max_contour, aspect_ratio_rotated, contour_area, rotated_rect_area



if __name__ == "__main__":
    try:
        mask = cv2.imread("mask/Scr_0020.jpg",cv2.IMREAD_GRAYSCALE)
        max_contour, aspect_ratio, contour_area, rect_area = analyze_largest_white_contour(mask)
        print(f"最大轮廓面积：{contour_area:.2f} 像素")
        print(f"包围矩形长宽比（宽/高）：{aspect_ratio:.2f}")
        print(f"包围矩形面积：{rect_area:.2f} 像素")
        print(f"轮廓填充率（轮廓面积/矩形面积）：{contour_area/rect_area:.2f}")
    except Exception as e:
        print(f"错误：{e}")