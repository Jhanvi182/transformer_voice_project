# ----------------------------------------------------
# CLEAN STARTUP (remove TensorFlow warnings completely)
# ----------------------------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ----------------------------------------------------
# IMPORTS
# ----------------------------------------------------
import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
from scipy.io.wavfile import write
import torchaudio
import torch
from pydub import AudioSegment

from transformers import Wav2Vec2Processor
from model import VoiceClassifier

# ----------------------------------------------------
# DEVICE
# ----------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------
# LOAD MODEL & PROCESSOR
# ----------------------------------------------------
try:
    model = VoiceClassifier()
    state = torch.load("transformer_voice_model.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
except Exception as e:
    messagebox.showerror("Model Error", f"Could not load model:\n{e}")
    raise SystemExit

try:
    processor = Wav2Vec2Processor.from_pretrained("processor")
except Exception as e:
    messagebox.showerror("Processor Error", f"Could not load processor folder:\n{e}")
    raise SystemExit

AGE_MAP = ["<18", "18-50", ">50"]
GENDER_MAP = ["Male", "Female"]

# =====================================================
# Convert any audio → WAV
# =====================================================
def convert_to_wav(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return path
    try:
        audio = AudioSegment.from_file(path)
        out_path = "converted.wav"
        audio.export(out_path, format="wav")
        return out_path
    except:
        messagebox.showerror("File Error", "Unsupported or corrupted audio format.")
        return None

# =====================================================
# Prediction Function
# =====================================================
def predict_audio(path):
    try:
        path = convert_to_wav(path)
        if path is None:
            return None, None

        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)
        if wav.abs().max() == 0:
            raise ValueError("Audio is silent.")

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav / (wav.abs().max() + 1e-9)

        inputs = processor([wav.numpy()], sampling_rate=16000,
                           return_tensors="pt", padding=True)

        with torch.no_grad():
            g_out, a_out = model(inputs.input_values.to(DEVICE))

        return (
            GENDER_MAP[int(g_out.argmax().item())],
            AGE_MAP[int(a_out.argmax().item())],
        )
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Error analyzing audio:\n{e}")
        return None, None

# =====================================================
# Session History Storage
# =====================================================
history_data = []

def update_history(filename, gender, age):
    entry = f"{filename} → {gender}, {age}"
    history_data.append(entry)
    history_box.insert(tk.END, entry)
    history_box.yview_moveto(1.0)

# =====================================================
# GUI ACTIONS
# =====================================================
def upload_file():
    path = filedialog.askopenfilename(
        filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All Files", "*.*")]
    )
    if not path:
        return
    result_label.config(text="Processing...", fg="#FFD54F")
    root.update_idletasks()
    gender, age = predict_audio(path)
    if gender:
        color = "#1B5E20" if gender == "Male" else "#D81B60"
        result_label.config(text=f"Gender: {gender}\nAge Group: {age}", fg=color)
        update_history(os.path.basename(path), gender, age)

def record_audio():
    try:
        duration = 3
        sample_rate = 16000
        out_file = "recorded.wav"
        sd.default.samplerate = sample_rate
        sd.default.channels = 1
        messagebox.showinfo("Recording", "Recording started. Speak now!")
        rec = sd.rec(int(duration * sample_rate), dtype='float32')
        sd.wait()
        write(out_file, sample_rate, rec)
        gender, age = predict_audio(out_file)
        if gender:
            color = "#1B5E20" if gender == "Male" else "#D81B60"
            result_label.config(text=f"Gender: {gender}\nAge Group: {age}", fg=color)
            update_history("Recorded Audio", gender, age)
    except Exception as e:
        messagebox.showerror("Recording Error", f"Error: {e}")

# =====================================================
# COMPACT MODERN UI
# =====================================================
root = tk.Tk()
root.title("🎤 Voice Gender & Age Classifier")
root.geometry("650x650")
root.config(bg="#E8FDF5")  # pastel green background

# Header
title = tk.Label(root, text="🎙 Voice Classifier", font=("Helvetica", 26, "bold"), bg="#E8FDF5", fg="#1B5E20")
title.pack(pady=(20,5))

subtitle = tk.Label(root, text="Upload or record your voice to predict Gender & Age", font=("Helvetica", 12), bg="#E8FDF5", fg="#555555")
subtitle.pack(pady=(0,15))

# Mic Button
mic_frame = tk.Frame(root, width=180, height=180, bg="#E8FDF5")
mic_frame.pack()
mic_btn = tk.Button(mic_frame, text="🎤", font=("Helvetica", 32, "bold"), bg="#1B5E20", fg="white",
                    activebackground="#4CAF50", activeforeground="white", relief="raised",
                    bd=4, command=record_audio, cursor="hand2")
mic_btn.place(x=10, y=10, width=160, height=160)

mic_label = tk.Label(root, text="Tap Mic to Record", font=("Helvetica", 11), bg="#E8FDF5", fg="#555555")
mic_label.pack(pady=10)

# Upload Button
upload_btn = tk.Button(root, text="📁 Upload Audio File", font=("Helvetica", 12, "bold"),
                       bg="#FFC107", fg="white", activebackground="#FFA000", activeforeground="white",
                       relief="flat", padx=15, pady=8, command=upload_file, cursor="hand2")
upload_btn.pack(pady=10)

# Result Panel
result_frame = tk.Frame(root, bg="white", bd=1, relief="groove")
result_frame.pack(pady=10, ipadx=8, ipady=8)
result_label = tk.Label(result_frame, text="Prediction will appear here", font=("Helvetica", 16, "bold"), bg="white", fg="#1B5E20")
result_label.pack(padx=15, pady=15)

# History Panel
history_title = tk.Label(root, text="📜 Session History", font=("Helvetica", 14, "bold"), bg="#E8FDF5", fg="#1B5E20")
history_title.pack(pady=(10,5))

history_frame = tk.Frame(root, bg="#E8FDF5")
history_frame.pack(pady=5)
history_box = tk.Listbox(history_frame, width=55, height=6, font=("Consolas", 11), bg="#E0F7FA", fg="#00796B", selectbackground="#B2DFDB")
history_box.pack(side=tk.LEFT, fill=tk.BOTH, padx=(10,0))
scroll = tk.Scrollbar(history_frame)
scroll.pack(side=tk.RIGHT, fill=tk.Y)
history_box.config(yscrollcommand=scroll.set)
scroll.config(command=history_box.yview)

# Footer
footer = tk.Label(root, text="💡 Powered by Wav2Vec2 Transformer", font=("Helvetica", 9, "italic"), bg="#E8FDF5", fg="#00796B")
footer.pack(side="bottom", pady=8)

# Run UI
root.mainloop()
