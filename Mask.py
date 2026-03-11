import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import cv2
import numpy as np


def preprocess_image_dynamic(img_path):
    """动态调整中值滤波窗口的预处理函数"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur = cv2.medianBlur(gray, 3)
    # cv2.imshow("med", blur)
    edges = cv2.Canny(gray, 20, 65)
    cv2.imshow("edges", edges)
    kernel = np.ones((7, 7), np.uint8)
    boundary_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4)

    boundary_closed = cv2.morphologyEx(boundary_closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cv2.imshow("edges2", boundary_closed)
    boundary = cv2.GaussianBlur(boundary_closed, (3, 3), 1)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=27,
        C=9
    )
    cv2.imshow("auto", thresh)
    (x1, y1, x2, y2) = crop_valid_white_area(
        img=thresh,
        noise_area_threshold=25,
        expand_pixel=30
    )
    contours, hierarchy = cv2.findContours(
        thresh,
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
    proportion = contour_area/rotated_rect_area
    perimeter = cv2.arcLength(max_contour, True)  # 轮廓周长
    circularity = (4 * np.pi * contour_area) / (perimeter ** 2)
    # 旋转矩形的长宽比（统一按 长/宽 计算，确保大于等于1）
    aspect_ratio_rotated = max(w, h) / min(w, h) if min(w, h) != 0 else 1.0
    print(max_contour, aspect_ratio_rotated, contour_area, rotated_rect_area,proportion,circularity)
    grade = average_grade(gray)
    edges = cv2.Canny(gray, 9, 20)
    cv2.imshow("Canny", edges)
    edges[y1:y2, x1:x2] = 0
    canny_edges = cv2.Canny(gray, 8, 24)
    cv2.imshow("Canny2",canny_edges)

    # 对比度增强
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # enhanced = clahe.apply(filtered_img)
    return img, gray


def average_grade(roi):
    dx = cv2.Sobel(roi, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    dy = cv2.Sobel(roi, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)

    # 5. 计算梯度幅值（两种方式二选一，与cv2.Canny的L1/L2gradient对应）
    # 方式1：L1范数（|dx| + |dy|，计算更快，对应cv2.Canny默认值）
    gradient_magnitude_L1 = np.abs(dx) + np.abs(dy)
    # 方式2：L2范数（√(dx² + dy²)，精度更高，对应cv2.Canny(L2gradient=True)）
    gradient_magnitude_L2 = np.sqrt(np.square(dx) + np.square(dy))

    # 6. 计算平均梯度（对梯度幅值求全局平均值）
    avg_gradient_L1 = np.mean(gradient_magnitude_L1)
    avg_gradient_L2 = np.mean(gradient_magnitude_L2)
    return avg_gradient_L1


def get_defect_boundary(enhanced_img):
    thresh = cv2.adaptiveThreshold(
        enhanced_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21,
        C=9
    )
    cv2.imshow("AutoThreshold (Defect Shape)", thresh)
    # 2. 步骤1：获取自适应阈值分割的缺陷主轮廓 + 计算minAreaRect

    (x1, y1, x2, y2) = crop_valid_white_area(
        img=thresh,
        noise_area_threshold=55,
        expand_pixel=20
    )
    roi = enhanced_img[y1:y2, x1:x2]
    grade = average_grade(roi)
    roi_edges = cv2.Canny(roi, 0.9 * grade, 1.6 * grade)

    cv2.imshow("Original Canny (With Noise)", roi_edges)
    edges_filtered = bitwise_or_in_roi(thresh, roi_edges, (x1, y1, x2, y2))
    cv2.imshow("Original Canny ", edges_filtered)

    kernel = np.ones((7, 7), np.uint8)
    boundary_closed = cv2.morphologyEx(edges_filtered, cv2.MORPH_CLOSE, kernel,iterations=4)
    # cv2.imshow("closed", boundary_closed)
    # 5. （可选）开运算过滤杂点 + 轮廓筛选（保持原有优化）
    boundary_closed = cv2.morphologyEx(boundary_closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    boundary = cv2.GaussianBlur(boundary_closed, (3, 3), 1)
    _, boundary = cv2.threshold(boundary, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("close + open", boundary)
    clean_boundary = np.zeros_like(boundary)
    for cnt in contours:
        if cv2.contourArea(cnt) > 230:
            cv2.drawContours(clean_boundary, [cnt], 0, 255, thickness=2)
    cv2.imshow("result",clean_boundary)
    return clean_boundary


def crop_valid_white_area(img, noise_area_threshold=50, expand_pixel=5):
    # 创建空白图像，用于绘制过滤噪声后的有效白色区域
    valid_white_img = np.zeros_like(img)

    # 2. 第一步：过滤白点噪声（保留大面积白色区域，剔除小噪声）
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > noise_area_threshold:
            cv2.drawContours(valid_white_img, [cnt], 0, 255, -1)  # 填充有效区域

    cv2.imshow("filter", valid_white_img)

    valid_white_pixels = np.argwhere(valid_white_img == 255)
    if len(valid_white_pixels) == 0:
        print("过滤噪声后无有效白色区域！")
        return img

    # 4. 第三步：计算有效白色区域的包围框
    y_coords = valid_white_pixels[:, 0]
    x_coords = valid_white_pixels[:, 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    # 5. 第四步：裁剪图像（可选扩大边界，避免裁到目标区域边缘）
    x_min = max(0, x_min - expand_pixel)
    y_min = max(0, y_min - expand_pixel)
    x_max = min(img.shape[1] - 1, x_max + expand_pixel)
    y_max = min(img.shape[0] - 1, y_max + expand_pixel)
    cropped_img = img[y_min:y_max, x_min:x_max]  # 从原图裁剪（保留原始细节）
    cv2.imshow("Cropped Valid Area", cropped_img)
    return (x_min, y_min, x_max, y_max)


# def bitwise_or_in_roi(bin_img1, bin_img2, roi_rect):
#     """
#     在指定区域内将相并两张二值图（区域外保持bin_img1不变）
#     :param bin_img1: 第一张二值图（背景图）
#     :param bin_img2: 第二张二值图（待相并的图）
#     :param roi_rect: 指定区域的矩形 [x_min, y_min, x_max, y_max]
#     :return: 仅指定区域相并后的结果图
#     """
#     # 1. 解析指定区域的坐标
#     x_min, y_min, x_max, y_max = roi_rect
#     # 确保坐标在图像范围内
#     x_min = max(0, x_min)
#     y_min = max(0, y_min)
#     x_max = min(bin_img1.shape[1]-1, x_max)
#     y_max = min(bin_img1.shape[0]-1, y_max)
#
#     # 2. 创建指定区域的掩码（仅指定区域为255，其余为0）
#     mask = np.zeros_like(bin_img1)
#     mask[y_min:y_max, x_min:x_max] = 255  # 矩形区域设为有效
#
#     # 3. 提取两张图在指定区域内的内容
#     img1_roi = cv2.bitwise_and(bin_img1, mask)
#     img2_roi = cv2.bitwise_and(bin_img2, mask)
#
#     # 4. 在指定区域内将相并两张图的内容
#     roi_merged = cv2.bitwise_or(img1_roi, img2_roi)
#
#     mask_final = np.zeros_like(bin_img1)
#
#     # 5. 将相并结果合并回原图（区域外保持bin_img1不变）
#     # 区域外：取bin_img1的内容；区域内：取相并后的内容
#     result = cv2.bitwise_or(
#         cv2.bitwise_and(mask_final, cv2.bitwise_not(mask)),  # 区域外：原图
#         roi_merged  # 区域内：相并结果
#     )
#     # cv2.imshow("bitwise_or_in_roi",result)
#     return result
def bitwise_or_in_roi(bin_img1, bin_img2, roi_rect):
    """
    在指定区域内将相并两张二值图（区域外保持bin_img1不变）
    :param bin_img1: 第一张二值图（背景图）
    :param bin_img2: 第二张二值图（待相并的图）
    :param roi_rect: 指定区域的矩形 [x_min, y_min, x_max, y_max]
    :return: 仅指定区域相并后的结果图
    """
    # 1. 解析指定区域的坐标
    x_min, y_min, x_max, y_max = roi_rect
    # 确保坐标在图像范围内
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(bin_img1.shape[1]-1, x_max)
    y_max = min(bin_img1.shape[0]-1, y_max)

    # 2. 创建指定区域的掩码（仅指定区域为255，其余为0）
    mask = np.zeros_like(bin_img1)
    mask[y_min:y_max, x_min:x_max] = 255  # 矩形区域设为有效

    # 3. 提取两张图在指定区域内的内容
    img_roi = cv2.bitwise_or(bin_img1[y_min:y_max,x_min:x_max], bin_img2)
    mask[y_min:y_max, x_min:x_max] = img_roi
    # cv2.imshow("bitwise_or_in_roi",result)
    return mask



def region_filling(boundary):
    filled = boundary.copy()
    # 找到所有轮廓，只对面积大的轮廓做填充
    contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 65:  # 只处理大轮廓（缺陷）
            seed_point = get_valid_seed_point(cnt, filled)
            if seed_point is not None:
                # 漫水填充
                cv2.floodFill(filled, None, seed_point, 255, flags=cv2.FLOODFILL_FIXED_RANGE | 8)
                cv2.imshow("valid_seed_filled", filled)
    # 生成最终mask：合并填充区域和缺陷边界（确保边界+内部都完整）
    final_mask = cv2.bitwise_or(filled, boundary)
    cv2.imshow("final_mask",final_mask)
    return filled, final_mask  # 修正：返回2个值，匹配调用时的接收变量


def get_valid_seed_point(contour, img):
    """
    从轮廓内部黑色区域获取有效种子点
    :param contour: 单个轮廓（cnt）
    :param img: 二值图像（0=黑色，255=白色）
    :return: 有效种子点(x, y)，无有效点返回None
    """
    # 1. 获取轮廓的包围矩形（缩小采样范围，提高效率）
    x, y, w, h = cv2.boundingRect(contour)
    # 包围矩形边界（避免超出图像范围）
    img_h, img_w = img.shape
    x_min = max(0, x)
    y_min = max(0, y)
    x_max = min(img_w - 1, x + w)
    y_max = min(img_h - 1, y + h)

    # 2. 随机采样+双重验证（最多采样100次，避免死循环）
    for _ in range(100):
        # 在包围盒内随机生成候选点
        cx = random.randint(x_min, x_max)
        cy = random.randint(y_min, y_max)
        candidate_point = (cx, cy)

        # 验证1：点是否在轮廓内部（cv2.pointPolygonTest 核心用法）
        # 返回值：>0 内部；=0 轮廓上；<0 外部
        inside_contour = cv2.pointPolygonTest(contour, candidate_point, measureDist=False)
        # 验证2：点是否是黑色像素（0）
        is_black = (img[cy, cx] == 0)

        # 两个条件都满足，返回该点
        if inside_contour > 0 and is_black:
            return candidate_point

    # 采样100次无有效点，返回None
    return None


def visualize_result(img, boundary, filled_region, final_mask):
    """可视化区域填充过程"""
    plt.figure(figsize=(16, 8))
    # 原始图像
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    # 缺陷边界
    plt.subplot(2, 2, 2)
    plt.imshow(boundary, cmap='gray')
    plt.title('缺陷边界')
    plt.axis('off')
    # 填充后的区域
    plt.subplot(2, 2, 3)
    plt.imshow(filled_region, cmap='gray')
    plt.title('区域填充结果')
    plt.axis('off')
    # 最终mask
    plt.subplot(2, 2, 4)
    plt.imshow(final_mask, cmap='gray')
    plt.title('最终完整mask')
    plt.axis('off')
    plt.tight_layout()
    plt.show()




# 测试入口
if __name__ == "__main__":
    img_path = "result/Oil_0024.jpg"  # 替换为你的图路径
    try:
        img, gray = preprocess_image_dynamic(img_path)
        boundary = get_defect_boundary(gray)
        filled_region, final_mask = region_filling(boundary)  # 现在返回2个值，匹配接收
        visualize_result(img, boundary, filled_region, final_mask)
        cv2.imwrite("区域填充最终mask.png", final_mask)
        print("区域填充mask已保存！")
    except Exception as e:
        print(f"出错：{e}")