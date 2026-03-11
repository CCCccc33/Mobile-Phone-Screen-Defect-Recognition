import cv2
import numpy as np
import os
import re



def background_segment(img):
    # 1. 读取图像并转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2. 阈值分割：目标是深色，背景是较亮的区域 → 用反向二值化提取目标
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)  # 深色区域转为白色
    # 3. 查找轮廓：找到目标的最大轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)  # 最大轮廓对应目标区域
    # 4. 计算最小外接矩形（即目标的最小内接矩形）
    min_rect = cv2.minAreaRect(max_contour)  # 返回 (中心(x,y), (宽,高), 旋转角度)
    rect_points = cv2.boxPoints(min_rect)  # 转换为四个顶点坐标
    rect_points = np.int0(rect_points)  # 转为整数
    sorted = sort_coordinates(rect_points)
    x_sorted = sorted[:, 0]  # 所有行的第0列（横坐标）
    y_sorted = sorted[:, 1]  # 所有行的第1列（纵坐标）
    # print(rect_points)
    return (x_sorted[1],y_sorted[1],x_sorted[2],y_sorted[2])


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


def calculate_iou(bbox1, bbox2):
    """
    计算两个边界框的交并比（IoU）
    bbox格式：(x1, y1, x2, y2)
    """
    # 提取坐标
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # 计算交集区域的坐标
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    # 计算交集面积
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # 计算两个框的面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # 计算并集面积
    union_area = area1 + area2 - inter_area

    # 计算IoU（避免除零错误）
    if union_area == 0:
        return 0.0
    iou = inter_area / union_area
    return iou

def get_number_from_filename(filename):
    """
    从文件名中提取纯数字（适配Oil_0033.jpg格式，提取0033或33）
    返回：整数类型的编号（如33，自动去除前导零，也可保留字符串格式）
    """
    # 正则匹配所有数字字符
    num_str = ''.join(re.findall(r'\d+', filename))
    if not num_str:
        return None
    # 转换为整数（自动去除前导零，若需保留前导零，直接返回num_str即可）
    return int(num_str)

def process_images_by_numbers(folder_path, target_numbers):
    """
    根据指定的数字编号筛选图片并处理，找到最小IoU对应的屏幕位置
    参数：
        folder_path: 图像文件夹路径
        target_numbers: 要筛选的编号列表（如[33, 35, 40]，对应Oil_0033.jpg等）
    返回：
        min_iou_value: 最小IoU值
        min_iou_bbox_pair: 最小IoU对应的两个屏幕bbox
        stable_min_bbox: 两个屏幕的交集区域（最小重叠位置）
        filtered_bboxes: 筛选出的所有图片的屏幕bbox字典（文件名: bbox）
    """
    filtered_bboxes = {}  # 存储筛选后的文件名与对应屏幕bbox
    target_numbers_set = set(target_numbers)  # 转为集合提高查找效率

    # 第一步：根据编号筛选图片并检测屏幕bbox
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # 跳过非图像文件

        # 提取文件名中的数字编号
        file_number = get_number_from_filename(filename)
        if file_number is None:
            print(f"文件名 {filename} 中未提取到数字，跳过")
            continue

        # 判断是否在目标编号列表中
        if file_number in target_numbers_set:
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图像 {filename}，跳过")
                continue

            # 检测屏幕bbox
            screen_bbox = background_segment(img)
            if screen_bbox is not None:
                filtered_bboxes[filename] = screen_bbox
                print(f"已筛选并检测 {filename}，屏幕位置：{screen_bbox}")
            else:
                print(f"未在 {filename} 中检测到屏幕，跳过")

    # 检查筛选后的有效图像数量
    if len(filtered_bboxes) < 2:
        raise ValueError(f"筛选后有效屏幕图像数量为 {len(filtered_bboxes)}，不足2张无法计算IoU")

    # 提取筛选后的bbox列表和文件名列表
    filenames = list(filtered_bboxes.keys())
    bboxes = list(filtered_bboxes.values())

    # 第二步：计算两两bbox的IoU，找到最小IoU
    min_iou_value = 1.0
    min_iou_bbox_pair = None
    min_iou_filename_pair = None

    for i in range(len(bboxes)):
        bbox1 = bboxes[i]
        fname1 = filenames[i]
        for j in range(i + 1, len(bboxes)):
            bbox2 = bboxes[j]
            fname2 = filenames[j]

            current_iou = calculate_iou(bbox1, bbox2)
            # 更新最小IoU信息
            if current_iou < min_iou_value:
                min_iou_value = current_iou
                min_iou_bbox_pair = (bbox1, bbox2)
                min_iou_filename_pair = (fname1, fname2)

    # 第三步：计算最小IoU对应的交集区域（稳定最小屏幕位置）
    bbox1, bbox2 = min_iou_bbox_pair
    fname1, fname2 = min_iou_filename_pair

    inter_x1 = max(bbox1[0], bbox2[0])
    inter_y1 = max(bbox1[1], bbox2[1])
    inter_x2 = min(bbox1[2], bbox2[2])
    inter_y2 = min(bbox1[3], bbox2[3])
    stable_min_bbox = (inter_x1, inter_y1, inter_x2, inter_y2)

    # 输出结果
    print("\n" + "="*60)
    print(f"筛选编号：{target_numbers}")
    print(f"最小IoU值：{min_iou_value:.6f}")
    print(f"最小IoU对应图像：{fname1} 和 {fname2}")
    print(f"{fname1} 屏幕位置：{bbox1}")
    print(f"{fname2} 屏幕位置：{bbox2}")
    print(f"稳定交集区域（最小重叠屏幕位置）：{stable_min_bbox}")
    print(f"交集区域面积：{(stable_min_bbox[2]-stable_min_bbox[0])*(stable_min_bbox[3]-stable_min_bbox[1])} 像素")
    print("="*60)

    return min_iou_value, min_iou_bbox_pair, stable_min_bbox, filtered_bboxes

# 调用示例
if __name__ == "__main__":
    # 配置参数
    image_folder = r"Scr"  # 替换为你的图像文件夹路径
    # 要筛选的编号列表（对应Oil_0033.jpg的33、Oil_0035.jpg的35等）
    target_image_numbers = [1,2,13,24,26,27,28,29,30,31,32,33]  # 可根据你的需求修改

    try:
        min_iou, bbox_pair, stable_bbox, filtered_dict = process_images_by_numbers(
            image_folder, target_image_numbers
        )
        print("\n处理完成！")
    except Exception as e:
        print(f"错误信息：{e}")