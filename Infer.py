from pipeline import build_audiosep, inference
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = build_audiosep(
    config_yaml='config/audiosep_base.yaml',
    checkpoint_path='checkpoint/audiosep_base_4M_steps.ckpt',
    device=device)

# 选择一个音频文件进行测试
audio_file = 'ESC-50-master/audio/1-19898-C-41.wav'  # ESC-50 数据集中的一个示例音频文件
text = 'motorcycle'  # 你可以根据数据集的标签来选择合适的描述
output_file = 'separated_audio.wav'  # 输出分离的音频文件

# AudioSep processes the audio at 32 kHz sampling rate
inference(model, audio_file, text, output_file, device)