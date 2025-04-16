import os

def split_file(filename, chunk_size_mb=50):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    chunk_size = chunk_size_mb * 1024 * 1024

    with open(file_path, 'rb') as f:
        i = 0
        while chunk := f.read(chunk_size):
            part_path = f"{file_path}.part_{i:03d}"
            with open(part_path, 'wb') as chunk_file:
                chunk_file.write(chunk)
            i += 1
    print(f"✅ 分割完成，共分成 {i} 个文件。")

# 使用示例
split_file("music_speech_audioset_epoch_15_esc_89.98.pt", chunk_size_mb=50)
split_file("audiosep_base_4M_steps.ckpt", chunk_size_mb=50)