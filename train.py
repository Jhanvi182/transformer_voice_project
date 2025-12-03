import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor
from model import VoiceClassifier

# ------------------------------
# CONFIG
# ------------------------------
DATA_DIR = "data/train"
CSV_PATH = "data/labels.csv"
MODEL_SAVE = "transformer_voice_model.pt"
PROCESSOR_SAVE = "processor"

BATCH_SIZE = 8
EPOCHS = 6
LR = 3e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")


# ------------------------------
# DATASET
# ------------------------------
class VoiceDataset(Dataset):
    def __init__(self, csv_path, audio_dir):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir

    def augment(self, wav):
        if np.random.rand() < 0.5:
            wav *= np.random.uniform(0.8, 1.2)
        if np.random.rand() < 0.3:
            wav += np.random.randn(len(wav)) * 0.005
        return np.clip(wav, -1, 1)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row["file_name"])

        wav, sr = torchaudio.load(audio_path)
        wav = wav.mean(0)

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        if wav.abs().max() > 0:
            wav = wav / wav.abs().max()

        wav = wav.numpy().astype(np.float32)

        if len(wav) < 16000:
            wav = np.pad(wav, (0, 16000 - len(wav)))

        wav = self.augment(wav)

        gender = 0 if row["gender"].lower() == "male" else 1

        # age groups
        a = int(row["age"])
        age = 0 if a < 18 else 1 if a < 50 else 2

        return wav, gender, age

    def __len__(self):
        return len(self.df)


def collate_fn(batch):
    wavs, genders, ages = zip(*batch)
    inputs = processor(list(wavs), sampling_rate=16000,
                       return_tensors="pt", padding=True)
    return {
        "input_values": inputs.input_values,
        "gender": torch.tensor(genders),
        "age": torch.tensor(ages)
    }


# ------------------------------
# CLASS WEIGHTS (SAFE)
# ------------------------------
def safe_weights(df):
    def safe_div(total, count):
        return 1.0 if count == 0 else total / count

    # Gender
    gcount = df["gender"].value_counts().to_dict()
    male = gcount.get("male", 0)
    female = gcount.get("female", 0)
    total_g = male + female

    gender_w = torch.tensor([
        safe_div(total_g, male),
        safe_div(total_g, female)
    ], dtype=torch.float32).to(DEVICE)

    # Age
    df["age_class"] = pd.cut(df["age"], bins=[0, 18, 50, 200], labels=[0, 1, 2])
    ac = df["age_class"].value_counts().to_dict()

    a0 = ac.get(0, 0)
    a1 = ac.get(1, 0)
    a2 = ac.get(2, 0)
    total_a = a0 + a1 + a2

    age_w = torch.tensor([
        safe_div(total_a, a0),
        safe_div(total_a, a1),
        safe_div(total_a, a2)
    ], dtype=torch.float32).to(DEVICE)

    return gender_w, age_w


# ------------------------------
# TRAINING LOOP
# ------------------------------
def train():
    df = pd.read_csv(CSV_PATH)
    gender_w, age_w = safe_weights(df)

    dataset = VoiceDataset(CSV_PATH, DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=collate_fn)

    model = VoiceClassifier(freeze_encoder=True).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # FIXED scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=5
    )

    loss_gender = torch.nn.CrossEntropyLoss(weight=gender_w)
    loss_age = torch.nn.CrossEntropyLoss(weight=age_w)

    for epoch in range(1, EPOCHS + 1):

        # Unfreeze encoder gradually
        if epoch == 8:
            for p in model.encoder.parameters():
                p.requires_grad = True

        total_loss = 0

        for batch in loader:
            x = batch["input_values"].to(DEVICE)
            g = batch["gender"].to(DEVICE)
            a = batch["age"].to(DEVICE)

            optimizer.zero_grad()

            g_pred, a_pred = model(x)
            loss = loss_gender(g_pred, g) + loss_age(a_pred, a)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch}/{EPOCHS} — Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE)
    processor.save_pretrained(PROCESSOR_SAVE)


if __name__ == "__main__":
    train()
