# model.py
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class VoiceClassifier(nn.Module):
    """
    Model architecture matching the original training run:
    - Wav2Vec2 encoder
    - Multihead attention over encoder outputs
    - Fully connected stack with BatchNorm, ReLU and Dropout
    - Two heads: gender and age
    """
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        num_gender: int = 2,
        num_age: int = 3,
        freeze_encoder: bool = False
    ):
        super().__init__()
        # Pretrained encoder
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Attention layer (batch_first=True expects inputs: (batch, seq, embed))
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=4, batch_first=True)

        # Fully connected block (matches the old state_dict keys like fc.1, fc.4, fc.5, and BatchNorm)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 256),   # fc.0
            nn.BatchNorm1d(256),                # fc.1.* (running_mean/var present in old state)
            nn.ReLU(),                          # fc.2
            nn.Dropout(0.3),                    # fc.3
            nn.Linear(256, 64),                 # fc.4
            nn.BatchNorm1d(64),                 # fc.5.* (running_mean/var present)
            nn.ReLU()                           # fc.6
        )

        # Prediction heads
        self.gender_head = nn.Linear(64, num_gender)
        self.age_head = nn.Linear(64, num_age)

    def forward(self, input_values: torch.Tensor):
        """
        input_values: tensor shaped (batch, seq_len) or (batch, seq_len, 1) depending on processor
        returns: (gender_logits, age_logits)
        """
        # Encoder outputs : (batch, seq_len, hidden)
        enc_out = self.encoder(input_values).last_hidden_state

        # Self-attention (returns (batch, seq_len, hidden))
        attn_out, _ = self.attention(enc_out, enc_out, enc_out)

        # Pool over time dimension
        pooled = attn_out.mean(dim=1)   # shape: (batch, hidden)

        # FC block expects (batch, features) for BatchNorm1d
        features = self.fc(pooled)      # shape: (batch, 64)

        # Heads
        gender_logits = self.gender_head(features)
        age_logits = self.age_head(features)

        return gender_logits, age_logits
