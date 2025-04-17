import yaml
from typing import List
import torch
import numpy as np
import librosa
from scipy.io.wavfile import write
from utils import ignore_warnings, parse_yaml, load_ss_model
from models.clap_encoder import CLAP_Encoder
import argparse
import os
import json



def parse_music_tags(json_str):
    """解析音乐标签JSON字符串，返回可用于分离的标签列表"""
    try:
        # 解析JSON字符串
        music_info = json.loads(json_str)

        tags = []

        # 提取并处理音乐风格标签（以逗号分隔）
        if "music_genre" in music_info:
            genres = [genre.strip() for genre in music_info["music_genre"].split(",")]
            tags.extend(genres)

        # 提取并处理乐器标签
        if "instruments" in music_info:
            instruments = [inst.strip() for inst in music_info["instruments"].split(",")]
            tags.extend(instruments)

        # 添加其他可能有用的特征作为标签
        if "key" in music_info:
            tags.append(music_info["key"].strip())

        return tags

    except json.JSONDecodeError:
        print("无效的JSON格式")
        return []


# 使用示例
json_str = '{"music_genre": "rock, classic rock, indie rock, garage rock", "instruments": "guitar, bass, drums, voice", "key": "A minor", "time_signature": "4/4", "tempo": "62.517 bpm"}'
tags = parse_music_tags(json_str)
print(f"提取的标签: {tags}")

# 修改主程序中的标签处理部分
if __name__ == '__main__':
    # ... 原有的代码 ...

    # 将JSON风格信息转换为标签列表
    if args.text.startswith('{'):
        # 输入是JSON格式
        text_tags = parse_music_tags(args.text)
    else:
        # 输入是空格分隔的标签
        text_tags = args.text.split()

    print(f'分离音频的标签: {text_tags}')

    # 为每个标签创建对应的子目录
    for tag in text_tags:
        tag_dir = os.path.join(args.output_dir, tag)
        os.makedirs(tag_dir, exist_ok=True)

    # 使用提取的标签进行音频分离
    separate_audio(
        model=model,
        audio_file=args.audio_file,
        text_tags=text_tags,
        output_dir=args.output_dir,
        device=device,
        use_chunk=args.use_chunk
    )

def build_audiosep(config_yaml, checkpoint_path, device):
    ignore_warnings()
    configs = parse_yaml(config_yaml)

    query_encoder = CLAP_Encoder().eval()
    model = load_ss_model(configs=configs, checkpoint_path=checkpoint_path, query_encoder=query_encoder).eval().to(
        device)

    print(f'Loaded AudioSep model from [{checkpoint_path}]')
    return model


def separate_audio(model, audio_file, text_tags, output_dir, device='cuda', use_chunk=False):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载混合音频
    print(f'Loading audio from [{audio_file}]')
    mixture, fs = librosa.load(audio_file, sr=32000, mono=True)

    # 获取输入文件名（不含路径和扩展名）用于生成输出文件名
    base_filename = os.path.splitext(os.path.basename(audio_file))[0]

    # 处理每个文本标签
    for text in text_tags:
        print(f'Separating audio with textual query: [{text}]')

        with torch.no_grad():
            # 为当前文本标签创建条件
            conditions = model.query_encoder.get_query_embed(
                modality='text',
                text=[text],
                device=device
            )

            input_dict = {
                "mixture": torch.Tensor(mixture)[None, None, :].to(device),
                "condition": conditions,
            }

            # 执行分离
            if use_chunk:
                sep_segment = model.ss_model.chunk_inference(input_dict)
                sep_segment = np.squeeze(sep_segment)
            else:
                sep_segment = model.ss_model(input_dict)["waveform"]
                sep_segment = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()

            # 生成输出文件路径，使用文本标签作为文件名的一部分
            safe_text = text.replace(' ', '_').replace('/', '_').replace('\\', '_')
            output_file = os.path.join(output_dir, f"{base_filename}_{safe_text}.wav")

            # 保存分离后的音频
            write(output_file, 32000, np.round(sep_segment * 32767).astype(np.int16))
            print(f'Separated audio for "{text}" written to [{output_file}]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AudioSep Audio Separation')
    parser.add_argument('--audio_file', type=str, help='Path to the input audio file')
    parser.add_argument('--text', type=str, help='Textual queries for audio separation (space-separated)')
    parser.add_argument('--output_dir', type=str, help='Directory for output separated audio files')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--config_yaml', type=str, default='config/audiosep_base.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint/audiosep_base_4M_steps.ckpt',
                        help='Path to the model checkpoint')
    parser.add_argument('--use_chunk', action='store_true', help='Use chunk inference')
    parser.add_argument('--prompt',  type=str, help='Use original music prompt')

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = build_audiosep(
        config_yaml=args.config_yaml,
        checkpoint_path=args.checkpoint_path,
        device=device)


    # 正确地只调用一次separate_audio，传入所有标签
    if args.prompt:
        text_tags =[]
        text_tags.append(args.prompt)

        separate_audio(
            model=model,
            audio_file=args.audio_file,
            text_tags=text_tags,
            output_dir=args.output_dir,
            device=device,
            use_chunk=args.use_chunk
        )
    else:
        # 将文本参数按空格分割成标签列表
        # 将JSON风格信息转换为标签列表
        if args.text.startswith('{'):
            # 输入是JSON格式
            text_tags = parse_music_tags(args.text)
        else:
            # 输入是空格分隔的标签
            text_tags = args.text.split()

        print(f'分离音频的标签: {text_tags}')
        separate_audio(
            model=model,
            audio_file=args.audio_file,
            text_tags=text_tags,
            output_dir=args.output_dir,
            device=device,
            use_chunk=args.use_chunk
        )