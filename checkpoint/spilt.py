def split_file(file_path, chunk_size_mb=100):
    chunk_size = chunk_size_mb * 1024 * 1024
    with open(file_path, 'rb') as f:
        i = 0
        while chunk := f.read(chunk_size):
            with open(f"{file_path}.part_{i:03d}", 'wb') as chunk_file:
                chunk_file.write(chunk)
            i += 1
    print(f"✅ 分割完成，共分成 {i} 个文件。")

# 使用示例
split_file("checkpoint/music_speech_audioset_epoch_15_esc_89.98.zip", chunk_size_mb=50)
split_file("checkpoint/audiosep_base_4M_steps.ckpt", chunk_size_mb=50)