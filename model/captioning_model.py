import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feature_dim: int,
        word_embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        pad_idx: int = 0,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, word_embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=word_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.init_h = nn.Linear(feature_dim, hidden_dim * num_layers)
        self.init_c = nn.Linear(feature_dim, hidden_dim * num_layers)
        self.dropout = nn.Dropout(dropout)

    def _init_hidden(self, features: torch.Tensor):
        bsz = features.size(0)
        h0 = self.init_h(features).view(bsz, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        c0 = self.init_c(features).view(bsz, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        return h0, c0

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        # Teacher forcing uses all tokens except final one as decoder input.
        embedded = self.embedding(captions[:, :-1])
        embedded = self.dropout(embedded)

        hidden = self._init_hidden(features)
        out, _ = self.lstm(embedded, hidden)
        logits = self.fc(out)
        return logits

    @torch.no_grad()
    def sample(
        self,
        features: torch.Tensor,
        start_idx: int,
        end_idx: int,
        max_len: int = 30,
    ):
        hidden = self._init_hidden(features)
        inputs = torch.full((features.size(0), 1), start_idx, dtype=torch.long, device=features.device)
        generated = []

        for _ in range(max_len):
            emb = self.embedding(inputs)
            out, hidden = self.lstm(emb, hidden)
            logits = self.fc(out[:, -1, :])
            next_token = torch.argmax(logits, dim=-1)
            generated.append(next_token)
            inputs = next_token.unsqueeze(1)

        seq = torch.stack(generated, dim=1)

        final = []
        for row in seq:
            tokens = []
            for tok in row.tolist():
                if tok == end_idx:
                    break
                tokens.append(tok)
            final.append(tokens)
        return final


class FeatureCaptioningModel(nn.Module):
    """Captioning model that consumes pre-extracted CNN features."""

    def __init__(
        self,
        vocab_size: int,
        feature_dim: int,
        word_embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.decoder = DecoderRNN(
            vocab_size=vocab_size,
            feature_dim=feature_dim,
            word_embed_dim=word_embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            pad_idx=pad_idx,
        )

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        return self.decoder(features, captions)

    @torch.no_grad()
    def generate_from_features(
        self,
        features: torch.Tensor,
        start_idx: int,
        end_idx: int,
        max_len: int = 30,
    ):
        return self.decoder.sample(features, start_idx=start_idx, end_idx=end_idx, max_len=max_len)
