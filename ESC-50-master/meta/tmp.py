import csv
import json
import os


def csv_to_json(csv_file_path, json_file_path, audio_base_path):
    """
    将CSV文件转换为JSON文件。

    参数:
        csv_file_path: CSV文件路径
        json_file_path: 输出JSON文件路径
        audio_base_path: 音频文件的基路径（用于构造完整路径）
    """
    data = []

    # 读取CSV文件
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            # 提取需要的字段
            filename = row['filename']
            category = row['category']

            # 构造音频文件的完整路径
            wav_path = os.path.join(audio_base_path, filename)

            # 构造JSON数据结构
            data.append({
                "wav": wav_path,
                "caption": category
            })

    # 写入JSON文件
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump({"data": data}, json_file, indent=4)

    print(f"转换完成！JSON文件已保存到: {json_file_path}")


# 示例用法
if __name__ == "__main__":
    # CSV文件路径
    csv_file_path = "esc50.csv"

    # 输出JSON文件路径
    json_file_path = "output.json"

    # 音频文件的基路径（根据实际情况修改）
    audio_base_path = "./ESC-50-master/audio"  # 替换为音频文件的实际路径

    # 调用函数
    csv_to_json(csv_file_path, json_file_path, audio_base_path)