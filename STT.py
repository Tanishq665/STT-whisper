import torch
import torchaudio
import numpy as np
import argparse
import tkinter as tk
from tkinter import filedialog
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_audio_file():
    """Open a file dialog to select an audio file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.flac")])
    return file_path

def transcribe_audio(file_path, language="english"):
    """Load audio, process it with Whisper, and return transcription."""
    waveform, sampling_rate = torchaudio.load(file_path)
    waveform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(waveform)
    audio_array = waveform.numpy().squeeze()
    
    # Load Whisper model
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to(device)
    
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="translate")
    input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    
    # Generate transcription
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an MP3 file using Whisper model.")
    parser.add_argument("--file_path", type=str, help="Path to the MP3 file (or select interactively)")
    
    args = parser.parse_args()
    
    # If no file path is provided, open a file dialog
    if not args.file_path:
        print("Select an audio file to transcribe...")
        args.file_path = select_audio_file()

    if args.file_path:
        print(f"Processing file: {args.file_path}")
        transcription = transcribe_audio(args.file_path)
        print("\nTranscription:\n", transcription)
    else:
        print("No file selected. Exiting...")
