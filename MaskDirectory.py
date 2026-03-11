import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def analyze_largest_white_contour(gray_img):
    """
    从mask图像中提取最大白色轮廓，并计算其长宽比、轮廓面积、包围矩形面积

    参数:
        gray_img: 灰度mask图像（二值图像，白色为前景255，黑色为背景0）

    返回:
        aspect_ratio: 长宽比（宽/高）
        contour_area: 轮廓实际面积
        proportion: 轮廓面积/最小包围矩形面积
        circularity: 圆形度
    """
    blur = cv2.medianBlur(gray_img, 7)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=27,
        C=9
    )
    contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("mask图像中未检测到任何白色轮廓！")

    max_contour = max(contours, key=cv2.contourArea)
    feature = calculate_defect_features(max_contour, gray_img,"Pre")
    if feature is None:
        raise ValueError("最大轮廓特征计算失败！")
    return (feature["aspect_ratio_rotated"],
            feature["contour_area"],
            feature["proportion"],
            feature["circularity"])


def analyse_mask(gray):
    """掩码生成阶段的缺陷初判：快速区分光斑/其他"""
    try:
        (aspect_ratio, contour_area, proportion, circularity) = analyze_largest_white_contour(gray)
        if 100 < contour_area < 1200 and 0.5 < aspect_ratio < 1.75 and circularity > 0.8:
            return 9  # 9=光斑kernel
        elif 8 < aspect_ratio and proportion < 0.8:
            return 5  # 5=划痕kernel
        else:
            return 3  # 3=通用kernel
    except Exception as e:
        print(f"掩码初判出错：{e}")
        return 3


def preprocess_image(img):
    """预处理：保留更多细节（降低去噪强度）"""
    # 兼容灰度图/彩色图输入
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    kernel_size = analyse_mask(gray)
    # 中值滤波（ksize=1时不滤波）
    blur = cv2.medianBlur(gray, kernel_size)
    print(f"当前kernel为{kernel_size}")
    # 对比度增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    return enhanced, kernel_size


def get_defect_boundary(enhanced_img, kernel_size):
    thresh = cv2.adaptiveThreshold(
        enhanced_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=27,
        C=9
    )
    # 修复：crop_valid_white_area可能返回None，需处理
    roi_rect = crop_valid_white_area(
        img=thresh,
        noise_area_threshold=65,
        expand_pixel=30
    )
    if roi_rect is None:
        roi_rect = (0, 0, enhanced_img.shape[1] - 1, enhanced_img.shape[0] - 1)
    x1, y1, x2, y2 = roi_rect

    if kernel_size != 9:
        roi_edges = cv2.Canny(enhanced_img, 25, 65)
        edges_filtered = bitwise_or_in_roi(thresh, roi_edges, (x1, y1, x2, y2))
        kernel = np.ones((7, 7), np.uint8)
        boundary_closed = cv2.morphologyEx(edges_filtered, cv2.MORPH_CLOSE, kernel, iterations=4)
        # cv2.imshow("closed1",boundary_closed)
        boundary_closed = cv2.morphologyEx(boundary_closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # cv2.imshow("closed2", boundary_closed)
        boundary = cv2.GaussianBlur(boundary_closed, (3, 3), 1)
    else:
        boundary = thresh
    _, boundary = cv2.threshold(boundary, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_boundary = np.zeros_like(boundary)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            cv2.drawContours(clean_boundary, [cnt], 0, 255, thickness=2)
    filled, final_mask = region_filling(clean_boundary)
    return final_mask


def crop_valid_white_area(img, noise_area_threshold=50, expand_pixel=5):
    """裁剪有效白色区域，返回(x_min,y_min,x_max,y_max)，无有效区域返回None"""
    valid_white_img = np.zeros_like(img)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > noise_area_threshold:
            cv2.drawContours(valid_white_img, [cnt], 0, 255, -1)

    valid_white_pixels = np.argwhere(valid_white_img == 255)
    if len(valid_white_pixels) == 0:
        print("过滤噪声后无有效白色区域！")
        return None

    y_coords = valid_white_pixels[:, 0]
    x_coords = valid_white_pixels[:, 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    # 扩大边界（避免裁到目标区域边缘）
    x_min = max(0, x_min - expand_pixel)
    y_min = max(0, y_min - expand_pixel)
    x_max = min(img.shape[1] - 1, x_max + expand_pixel)
    y_max = min(img.shape[0] - 1, y_max + expand_pixel)

    return (x_min, y_min, x_max, y_max)


def bitwise_or_in_roi(bin_img1, bin_img2, roi_rect):
    """在指定区域内将相并两张二值图（区域外保持bin_img1不变）"""
    x_min, y_min, x_max, y_max = roi_rect
    # 确保坐标在图像范围内
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(bin_img1.shape[1] - 1, x_max)
    y_max = min(bin_img1.shape[0] - 1, y_max)

    # 创建指定区域的掩码
    mask = np.zeros_like(bin_img1)
    mask[y_min:y_max, x_min:x_max] = 255

    # 提取两张图在指定区域内的内容
    img1_roi = cv2.bitwise_and(bin_img1, mask)
    img2_roi = cv2.bitwise_and(bin_img2, mask)

    # 在指定区域内相并
    roi_merged = cv2.bitwise_or(img1_roi, img2_roi)

    # 合并回原图（区域外保持bin_img1不变）
    mask_final = np.zeros_like(bin_img1)
    result = cv2.bitwise_or(
        cv2.bitwise_and(bin_img1, cv2.bitwise_not(mask)),
        roi_merged
    )
    return result


def region_filling(boundary):
    """区域填充：从轮廓内部种子点漫水填充"""
    filled = boundary.copy()
    contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            seed_point = get_valid_seed_point(cnt, filled)
            if seed_point is not None:
                cv2.floodFill(filled, None, seed_point, 255, flags=cv2.FLOODFILL_FIXED_RANGE | 8)

    final_mask = cv2.bitwise_or(filled, boundary)
    return filled, final_mask


def get_valid_seed_point(contour, img):
    """从轮廓内部黑色区域获取有效种子点"""
    x, y, w, h = cv2.boundingRect(contour)
    img_h, img_w = img.shape
    x_min = max(0, x)
    y_min = max(0, y)
    x_max = min(img_w - 1, x + w)
    y_max = min(img_h - 1, y + h)

    # 随机采样（最多100次）
    for _ in range(100):
        cx = random.randint(x_min, x_max)
        cy = random.randint(y_min, y_max)
        candidate_point = (cx, cy)

        # 验证：点在轮廓内部且是黑色像素
        inside_contour = cv2.pointPolygonTest(contour, candidate_point, measureDist=False)
        is_black = (img[cy, cx] == 0)
        if inside_contour > 0 and is_black:
            return candidate_point

    return None


def white_pixel_count(contour, img):
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()
    contour_mask = np.zeros_like(gray_img, dtype=np.uint8)
    # 填充轮廓内部为白色（255），得到轮廓专属掩码
    cv2.drawContours(contour_mask, [contour], -1, 255, cv2.FILLED)
    img_in_contour = cv2.bitwise_and(gray_img, contour_mask)
    contour_area_pixel = np.sum(contour_mask == 255)
    # 步骤5：统计轮廓内灰度值>200的像素数
    img_white_count = np.sum(img_in_contour > 110)  # 110
    if contour_area_pixel == 0:
        print("警告：掩码内无有效白色区域")
        white_ratio = 0.0
    else:
        white_ratio = img_white_count / contour_area_pixel
    return white_ratio


def calculate_defect_features(contour, img, defect_type):
    """极致简写版（功能不变，代码更紧凑）"""
    area = cv2.contourArea(contour)
    if area < (10 if defect_type == "Pre" else 500):  # 500
        return None
    img_h, img_w = img.shape[:2]
    (cx, cy), (w, h), _ = cv2.minAreaRect(contour)
    rotated_rect_area, perimeter = w * h, cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    x, y, w_rect, h_rect = cv2.boundingRect(contour)  # x/y=左上角，w_rect/h_rect=宽高
    ar = max(w, h) / min(w, h) if min(w, h) != 0 else 1.0
    prop = area / rotated_rect_area if rotated_rect_area > 0 else 0
    white_ratio = white_pixel_count(contour,img)
    # 修正：判断框的完整上下左右边界，过滤所有边缘框
    edge_threshold = 65  # 边缘阈值，可调整
    if defect_type != "Pre" and defect_type != "Sta":
        if area < 740:
            return None
        # 上边缘：框顶部 < 阈值；下边缘：框底部 > 图像高度-阈值
        # 左边缘：框左侧 < 阈值；右边缘：框右侧 > 图像宽度-阈值
        if (y + h_rect / 2 < edge_threshold) or (y + h_rect / 2 > img_h - edge_threshold) or \
                (x + w_rect / 2 < edge_threshold) or (x + w_rect / 2 > img_w - edge_threshold):
            return None  # 过滤边缘框

    return {
        "aspect_ratio_rotated": ar, "contour_area": area, "proportion": prop,
        "circularity": circularity, "area": area, "aspect_ratio": ar,
        "centroid": (x + w_rect / 2, y + h_rect / 2), "bounding_rect": (x, y, w_rect, h_rect),
        "white_ratio": white_ratio
    }


def defect_classification(feature, preliminary_type):
    """融合掩码阶段初判的缺陷分类（统一返回小写+匹配统计字段）"""
    if feature is None:
        return "noise"

    area = feature["area"]
    aspect_ratio = feature["aspect_ratio"]
    circularity = feature["circularity"]
    proportion = feature["proportion"]
    white_ratio = feature["white_ratio"]

    # 融合初判结果
    if preliminary_type == "Sta":
        return "spot"
    elif preliminary_type == "Scr":
        if area >= 10000:
            return "oil"
        elif white_ratio > 0.0009:
            return "scratch"
        else:
            return "noise"
    elif preliminary_type == "Oil":
        if area < 800:
            return "scratch"
        elif white_ratio > 0.0009:
            return "oil"
        else:
            return "noise"
    else:
        if 100 < area < 1200 and 0.5 < aspect_ratio < 1.75 and circularity > 0.7:
            return "spot"
        elif 8 < aspect_ratio and proportion < 0.8 and 800 < area < 6000:
            return "scratch"
        elif area > 1000 and 0.5 < aspect_ratio < 3 and circularity > 0.3:
            return "oil"
        else:
            return "noise"


def calculate_gradient_features(img, contour):
    """计算缺陷区域的梯度特征：梯度均值、梯度方差、方向一致性"""
    # 1. 获取缺陷的ROI区域
    x, y, w, h = cv2.boundingRect(contour)
    roi = img[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0, 0.0, 0.0

    # 2. 计算Sobel梯度
    sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)  # 梯度幅值
    gradient_dir = np.arctan2(sobel_y, sobel_x)  # 梯度方向（弧度）

    # 3. 统计梯度特征
    mag_mean = np.mean(gradient_mag)  # 梯度幅值均值（油污：小）
    mag_var = np.var(gradient_mag)  # 梯度幅值方差（油污：小）
    # 方向一致性：计算方向的余弦和（越接近1，方向越一致）
    dir_cos_sum = np.mean(np.cos(gradient_dir - np.mean(gradient_dir)))

    return mag_mean, mag_var, dir_cos_sum


def calculate_freq_features(img, contour):
    """计算缺陷区域的频域特征：低频能量占比"""
    x, y, w, h = cv2.boundingRect(contour)
    roi = img[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0

    # 1. 傅里叶变换
    fft = np.fft.fft2(roi)
    fft_shift = np.fft.fftshift(fft)  # 低频移到中心
    magnitude = 20 * np.log(np.abs(fft_shift))  # 幅度谱（可视化用）

    # 2. 计算低频能量占比（中心区域能量 / 总能量）
    rows, cols = roi.shape
    # 取中心1/4区域为低频区（可调整）
    center_row, center_col = rows // 2, cols // 2
    low_freq_region = fft_shift[center_row - rows // 4:center_row + rows // 4,
                      center_col - cols // 4:center_col + cols // 4]
    low_energy = np.sum(np.abs(low_freq_region) ** 2)
    total_energy = np.sum(np.abs(fft_shift) ** 2)
    low_freq_ratio = low_energy / total_energy if total_energy > 0 else 0.0

    return low_freq_ratio


def detect_and_classify_defects(img, img_name, save_dir="defect_detection_results"):
    """端到端流程：生成掩码→提取特征→融合分类→可视化结果"""
    os.makedirs(save_dir, exist_ok=True)
    mask_path = os.path.join("mask",img_name)
    # 预处理
    enhanced, kernel_size = preprocess_image(img)
    final_mask = get_defect_boundary(enhanced, kernel_size)
    cv2.imwrite(mask_path, final_mask)
    # 初判类型映射（匹配defect_classification的输入）
    prelim_type_map = {9: "Sta", 5: "Scr", 3: "Oil"}
    preliminary_type = prelim_type_map.get(kernel_size, "Oil")

    img_copy = img.copy()
    if len(img_copy.shape) == 2:  # 灰度图转彩色图（方便标注彩色框）
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)

    # 初始化统计结果
    defect_results = {
        "total_defects": 0,
        "spot": 0,
        "scratch": 0,
        "oil": 0,
        "unknown": 0,
        "preliminary_type": preliminary_type,
        "defect_details": []
    }

    # 类别颜色映射
    color_map = {
        "spot": (0, 255, 0),  # 绿色：斑点
        "scratch": (0, 0, 255),  # 红色：划痕
        "oil": (255, 0, 0),  # 蓝色：油污
        "unknown": (128, 128, 128)  # 灰色：未知
    }
    contours, _ = cv2.findContours(
        final_mask,
        cv2.RETR_EXTERNAL,  # 只提取最外层轮廓
        cv2.CHAIN_APPROX_SIMPLE  # 压缩轮廓点
    )
    # 遍历轮廓分类
    for cnt in contours:
        feature = calculate_defect_features(cnt, img, preliminary_type)
        defect_type = defect_classification(feature, preliminary_type)

        if defect_type == "noise":
            continue

        # 更新统计
        defect_results["total_defects"] += 1
        if defect_type in defect_results:
            defect_results[defect_type] += 1

        # 保存缺陷详情
        defect_details = {
            "type": defect_type,
            "area": feature["area"],
            "aspect_ratio": feature["aspect_ratio"],
            "circularity": feature["circularity"],
            "centroid": feature["centroid"],
            "bounding_rect": feature["bounding_rect"]
        }
        defect_results["defect_details"].append(defect_details)

        # 可视化标注
        x, y, w, h = feature["bounding_rect"]
        color = color_map.get(defect_type, (128, 128, 128))
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)

        # 绘制重心点
        cx, cy = int(feature["centroid"][0]), int(feature["centroid"][1])
        cv2.circle(img_copy, (cx, cy), 3, color, -1)

        # 添加文字标注
        text = f"{defect_type} (area:{int(feature['area'])})"
        y_text = y - 10 if y - 10 > 0 else y + h + 20
        cv2.putText(
            img_copy, text, (x, y_text),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

    # 保存结果
    file_name = os.path.splitext(os.path.basename(img_name))[0]
    save_path = os.path.join(save_dir, f"{file_name}_detection.jpg")
    cv2.imwrite(save_path, img_copy)
    print(f"\n检测结果已保存至：{save_path}")

    # 打印统计
    print(f"【{file_name} 缺陷检测统计】")
    print(f"掩码阶段初判类型：{defect_results['preliminary_type']}")
    print(f"总缺陷数：{defect_results['total_defects']}")
    print(f"斑点：{defect_results['spot']} 个")
    print(f"划痕：{defect_results['scratch']} 个")
    print(f"油污：{defect_results['oil']} 个")
    print(f"未知缺陷：{defect_results['unknown']} 个")

    return defect_results


def feature_extract(folder_path):
    """遍历文件夹处理所有图像"""
    os.makedirs("mask", exist_ok=True)
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(folder_path, filename)
        # 读取彩色图（兼容后续预处理）
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像 {filename}，跳过")
            continue
        file = os.path.join("mask", filename)
        try:
            # enhanced, kernel_size = preprocess_image(img)
            # final_mask, contours = get_defect_boundary(enhanced, kernel_size)
            # cv2.imwrite(file,final_mask)
            detect_and_classify_defects(img, filename)
            print(f"{filename} 处理完成！")
        except Exception as e:
            print(f"处理 {filename} 出错：{e}")


