"""Self-supervised / audio-language pretrained baselines.

8. WavLM (Chen et al., IEEE JSTSP 2022) - fine-tuned microsoft/wavlm-base-plus,
   raw 16 kHz waveform input.
9. CLAP (Wu et al., ICASSP 2023) - laion/clap-htsat-unfused audio tower,
   linear probe on frozen embeddings (standard protocol).
"""
import torch
import torch.nn as nn


class WavLMClassifier(nn.Module):
    def __init__(self, num_classes=3, model_name="microsoft/wavlm-base-plus"):
        super().__init__()
        from transformers import WavLMModel
        self.backbone = WavLMModel.from_pretrained(model_name)
        self.backbone.feature_extractor._freeze_parameters()
        d = self.backbone.config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, num_classes))

    def forward(self, wave_i16):  # (B,L) int16 @16k
        # wavlm-base-plus ships do_normalize=false (group-norm feature
        # extractor), so raw [-1,1] waveform is the checkpoint's protocol
        x = wave_i16.float() / 32768.0
        h = self.backbone(x).last_hidden_state  # (B,T,D)
        return self.head(h.mean(dim=1))

    def param_groups(self, lr_backbone=3e-5, lr_head=1e-3):
        return [
            {"params": [p for p in self.backbone.parameters() if p.requires_grad], "lr": lr_backbone},
            {"params": self.head.parameters(), "lr": lr_head},
        ]


class CLAPEmbedder:
    """Frozen CLAP audio tower -> 512-d embeddings (computed once, cached)."""

    def __init__(self, device, model_name="laion/clap-htsat-unfused"):
        from transformers import ClapAudioModelWithProjection, ClapProcessor
        self.model = ClapAudioModelWithProjection.from_pretrained(model_name).to(device).eval()
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.device = device

    @torch.no_grad()
    def embed(self, waves_i16_48k, batch=32):
        out = []
        for i in range(0, len(waves_i16_48k), batch):
            chunk = [w.astype("float32") / 32768.0 for w in waves_i16_48k[i:i + batch]]
            try:
                inp = self.processor(audio=chunk, sampling_rate=48000, return_tensors="pt")
            except TypeError:
                inp = self.processor(audios=chunk, sampling_rate=48000, return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in inp.items()}
            e = self.model(**inp).audio_embeds  # (b,512)
            out.append(e.cpu())
        return torch.cat(out)


class LinearProbe(nn.Module):
    def __init__(self, dim=512, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.fc(x)
