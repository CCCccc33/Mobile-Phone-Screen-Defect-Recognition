import Edge
import MaskDirectory


# 运行代码（替换为你的图像路径）
if __name__ == "__main__":
    IMAGE_FILE_PATH = "defect_images"
    Edge.RemoveBackground(IMAGE_FILE_PATH)
    MaskDirectory.feature_extract("result")