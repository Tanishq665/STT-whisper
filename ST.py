import torch
import torchaudio
import numpy as np
import gradio as gr
from transformers import WhisperProcessor, WhisperForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to(device)

def transcribe_audio(file, language="english"):
    """Transcribes audio using Whisper."""
    waveform, sampling_rate = torchaudio.load(file)
    waveform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(waveform)
    audio_array = waveform.numpy().squeeze()
    
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="translate")
    input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription[0]

# Gradio Interface
demo = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Whisper Speech-to-Text",
    description="Upload an MP3/WAV audio file and get the transcribed text."
)

demo.launch()
