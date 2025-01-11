import argparse
import torch
import torchaudio
import numpy as np
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader
from nets.network_cnn import model
from utils import get_mean_and_std, image_loader

def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample_transform(waveform)
    # Convert to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform

def load_model(checkpoint_path, device):
    net = model(num_classes=4).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint)
    net.eval()
    return net

def infer(audio_path, model, device, transform):
    waveform = preprocess_audio(audio_path).to(device)
    waveform = transform(waveform)
    waveform = waveform.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(waveform)
        _, predicted = torch.max(output, 1)
    return predicted.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RespireNet: Lung Sound Inference')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to the lung sound audio file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the inference on')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    mean, std = [0.5091, 0.1739, 0.4363], [0.2288, 0.1285, 0.0743]
    transform = Compose([ToTensor(), Normalize(mean, std)])

    model = load_model(args.checkpoint, device)
    prediction = infer(args.audio_path, model, device, transform)
    print(f'Predicted class: {prediction}')