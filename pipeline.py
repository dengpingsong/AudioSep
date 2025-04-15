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

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = build_audiosep(
        config_yaml=args.config_yaml,
        checkpoint_path=args.checkpoint_path,
        device=device)

    # 将文本参数按空格分割成标签列表
    text_tags = args.text.split()
    print(f'Separating audio for tags: {text_tags}')

    # 正确地只调用一次separate_audio，传入所有标签
    separate_audio(
        model=model,
        audio_file=args.audio_file,
        text_tags=text_tags,
        output_dir=args.output_dir,
        device=device,
        use_chunk=args.use_chunk
    )