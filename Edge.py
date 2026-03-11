import cv2
import numpy as np
import os


def calculate_average_gray(img):
    """
    计算图片的平均灰度值
    参数:
        image_path: 图片文件路径（本地路径）或网络图片URL
    返回:
        float: 图片的平均灰度值（范围 0-255，0=纯黑，255=纯白）
    """
    # 2. 转换为灰度图（彩色图转灰度的标准做法）
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3. 计算平均灰度值（np.mean 计算所有像素的平均值）
    avg_gray = np.mean(gray_img)
    return round(avg_gray, 2)  # 保留两位小数，方便查看


def photo_classify(img):
    try:
        h, w = img.shape[:2]
        avg_gray = calculate_average_gray(img)
        print(f"平均灰度值：{avg_gray}")
        threshold = 65  # 阈值可根据你的样本调整
        if avg_gray < threshold:
            print("判断：图为全黑手机")
            """
            screen_x1 = 258   # 屏幕左边界
            screen_y1 = 200  # 屏幕上边界
            screen_x2 = 1673  # 屏幕右边界
            screen_y2 = 914  # 屏幕下边界
            """
            mask_calculate = background_segment(img, "black")
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = 258, 205, 1673, 905
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            intersection_mask = cv2.bitwise_and(mask_calculate, mask)
            return intersection_mask
        else:
            print("判断：图为带白色部分的手机")
            return background_segment(img, "white")
    except Exception as e:
        print(f"计算失败：{e}")


def background_segment(img,type):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if type == "white":
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    else:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)  # 最大轮廓对应目标区域
    min_rect = cv2.minAreaRect(max_contour)  # 返回 (中心(x,y), (宽,高), 旋转角度)
    rect_points = cv2.boxPoints(min_rect)  # 转换为四个顶点坐标
    rect_points = np.int0(rect_points)  # 转为整数
    mask = np.zeros((h, w), dtype=np.uint8)
    sorted = sort_coordinates(rect_points)
    x_sorted = sorted[:, 0]  # 所有行的第0列（横坐标）
    y_sorted = sorted[:, 1]  # 所有行的第1列（纵坐标）
    if type == "white":
        (x1, y1, x2, y2) = (x_sorted[1] + 4, y_sorted[1] + 3, x_sorted[2] - 4, y_sorted[2] - 4)
    else:
        (x1, y1, x2, y2) = (x_sorted[1] + 4, y_sorted[1] + 19, x_sorted[2] - 4, y_sorted[2]-4)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    result = img.copy()
    cv2.drawContours(result, [rect_points], 0, (0, 255, 0), 2)  # 绘制最小矩形
    cv2.imwrite("seg.jpg", result)
    return mask


def sort_coordinates(arr):
    """
    将numpy数组的横坐标和纵坐标分别按从小到大排序
    （默认输入为形状类似 (n, 2) 的数组，每一行是一个点[横坐标, 纵坐标]）

    参数:
        arr: numpy数组，每一行是一个点[横坐标, 纵坐标]

    返回:
        numpy数组: 横坐标和纵坐标分别升序排列后的数组
    """
    # 提取横坐标（第0列）和纵坐标（第1列）
    x_coords = arr[:, 0]  # 所有行的第0列（横坐标）
    y_coords = arr[:, 1]  # 所有行的第1列（纵坐标）

    # 分别对横、纵坐标进行升序排序
    x_sorted = np.sort(x_coords)
    y_sorted = np.sort(y_coords)
    # 将排序后的横、纵坐标重新组合成数组
    sorted_arr = np.column_stack((x_sorted, y_sorted))
    return sorted_arr


def RemoveBackground(folder_path):
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # 跳过非图像文件
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        if img is None:
            print(f"无法读取图像 {filename}，跳过")
            continue
        file = os.path.join("result", filename)
        os.makedirs("result", exist_ok=True)
        mask = photo_classify(img)
        non_zero_coords = cv2.findNonZero(mask)
        if non_zero_coords is not None:
            x, y, w, h = cv2.boundingRect(non_zero_coords)
            # 裁剪最小包围区域
            cropped_image = img[y:y + h, x:x + w]
            print(f"已裁剪无用黑边，裁剪区域：x={x}, y={y}, 宽={w}, 高={h}")
            cv2.imwrite(file, cropped_image)
        else:
            print("non_zero_coords is None")



