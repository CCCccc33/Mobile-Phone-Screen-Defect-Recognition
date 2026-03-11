import pandas as pd
import os


def txt_to_xlsx(txt_file_path, xlsx_save_path=None):
    """
    从TXT读取数据并写入XLSX
    :param txt_file_path: TXT文件路径（必填）
    :param xlsx_save_path: XLSX保存路径（默认保存在TXT同目录）
    """
    # 1. 处理保存路径
    if xlsx_save_path is None:
        # 获取TXT文件所在目录和文件名（去除后缀）
        txt_dir = os.path.dirname(txt_file_path)
        txt_name = os.path.splitext(os.path.basename(txt_file_path))[0]
        xlsx_save_path = os.path.join(txt_dir, f"{txt_name}_转换后.xlsx")

    # 2. 读取TXT数据（处理每行数值，兼容科学计数法）
    data_list = []
    with open(txt_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # 去除换行符和空格
            if line:  # 跳过空行
                # 转换为浮点数（自动识别科学计数法）
                try:
                    value = float(line)
                    data_list.append(value)
                except ValueError:
                    print(f"跳过无效数据行：{line}")

    # 3. 整理数据为DataFrame（添加索引列，便于查看顺序）
    df = pd.DataFrame({
        "序号": range(1, len(data_list) + 1),  # 从1开始的索引
        "原始数据": data_list
    })

    # 4. 写入XLSX文件
    df.to_excel(
        excel_writer=xlsx_save_path,
        sheet_name="TXT转换数据",
        index=False,  # 不写入pandas默认索引
        engine="openpyxl"  # 处理.xlsx格式的引擎
    )

    print(f"转换完成！XLSX文件保存路径：{xlsx_save_path}")
    print(f"共读取并保存 {len(data_list)} 条有效数据")
    return df  # 可选返回DataFrame，便于后续分析


# ------------------- 调用函数（修改TXT路径即可）-------------------
if __name__ == "__main__":
    # 请替换为你的TXT文件实际路径（绝对路径或相对路径都可）
    TXT_FILE_PATH = r"C:\Users\asus\Desktop\data\数据2-噪声大.txt"  # 若TXT和代码在同一文件夹，直接写文件名；否则写完整路径（如"C:/Users/xxx/数据2-噪声大.txt"）

    # 调用转换函数
    result_df = txt_to_xlsx(TXT_FILE_PATH)

    # 可选：打印前5条数据预览
    print("\n数据预览：")
    print(result_df.head())