import torch
import torchaudio
from transformers import Wav2Vec2Processor
from model import VoiceClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = VoiceClassifier()
model.load_state_dict(torch.load("transformer_voice_model.pt", map_location=DEVICE))
model.to(DEVICE).eval()

processor = Wav2Vec2Processor.from_pretrained("processor")

AGE_MAP = ["<18", "18-50", ">50"]
GENDER_MAP = ["Male", "Female"]

def predict(audio_path):
    wav, sr = torchaudio.load(audio_path)
    wav = wav.mean(dim=0)

    if wav.abs().max() == 0:
        print("Audio is silent!")
        return

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    wav = wav / (wav.abs().max() + 1e-9)

    inputs = processor(
        wav.numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():
        g_log, a_log = model(inputs.input_values.to(DEVICE))

    gender = GENDER_MAP[g_log.argmax().item()]
    age = AGE_MAP[a_log.argmax().item()]

    return gender, age
