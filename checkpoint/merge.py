import os

def merge_file(output_path, part_prefix):
    parts = sorted([f for f in os.listdir(os.path.dirname(part_prefix)) if f.startswith(os.path.basename(part_prefix))])
    with open(output_path, 'wb') as out_file:
        for part in parts:
            with open(os.path.join(os.path.dirname(part_prefix), part), 'rb') as p:
                out_file.write(p.read())
    print("✅ 合并完成：", output_path)

# 使用示例
merge_file("checkpoint/music_speech_audioset_epoch_15_esc_89.98.zip", "checkpoint/music_speech_audioset_epoch_15_esc_89.98.zip.part_")