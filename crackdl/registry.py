"""Model registry: name -> (builder, input_kind, train_config)."""
import torch


def build(name, num_classes=3):
    """Returns (model, cfg) where cfg has: input ('mel32'|'wave16'|'clap48'),
    epochs, batch_size, lr, weight_decay, mixup, specaug, param_groups(optional)."""
    if name == "cnn14":
        from models_cnn import CNN14
        return CNN14(num_classes), dict(input="mel32", epochs=60, batch_size=64,
                                        lr=3e-4, weight_decay=1e-4, mixup=0.2, specaug=True)
    if name == "bcresnet":
        from models_cnn import BCResNet
        return BCResNet(num_classes), dict(input="mel32", epochs=80, batch_size=64,
                                           lr=1e-3, weight_decay=1e-4, mixup=0.2, specaug=True)
    if name == "mobilenetv3":
        from models_cnn import build_mobilenetv3
        return build_mobilenetv3(num_classes), dict(input="mel32", epochs=60, batch_size=64,
                                                    lr=5e-4, weight_decay=1e-4, mixup=0.2, specaug=True)
    if name == "conformer":
        from models_transformer import ConformerClassifier
        return ConformerClassifier(num_classes), dict(input="mel32", epochs=80, batch_size=64,
                                                      lr=5e-4, weight_decay=1e-4, mixup=0.2, specaug=True)
    if name == "ast":
        from models_transformer import ASTClassifier
        return ASTClassifier(num_classes), dict(input="mel32", epochs=40, batch_size=64,
                                                lr=1e-4, weight_decay=1e-4, mixup=0.2, specaug=True)
    if name == "passt":
        from models_transformer import PaSSTClassifier
        return PaSSTClassifier(num_classes), dict(input="mel32", epochs=40, batch_size=64,
                                                  lr=1e-4, weight_decay=1e-4, mixup=0.2, specaug=True)
    if name == "htsat":
        from models_transformer import HTSATClassifier
        return HTSATClassifier(num_classes), dict(input="mel32", epochs=80, batch_size=64,
                                                  lr=5e-4, weight_decay=5e-2, mixup=0.2, specaug=True)
    if name == "wavlm":
        from models_ssl import WavLMClassifier
        m = WavLMClassifier(num_classes)
        return m, dict(input="wave16", epochs=15, batch_size=32, lr=3e-5,
                       weight_decay=1e-4, mixup=0.2, specaug=False,
                       param_groups=m.param_groups())
    if name == "audiomamba":
        from models_ssm import AudioMamba
        return AudioMamba(num_classes), dict(input="mel32", epochs=60, batch_size=64,
                                             lr=5e-4, weight_decay=5e-2, mixup=0.2, specaug=True)
    if name.startswith("roastformer"):
        from roastformer import RoastFormer
        # Final RoastFormer configuration (selected on validation macro-F1):
        # scaled backbone + acceleration (delta-delta) stream + SE channel gating.
        BASE = dict(width=(64, 96, 128), d=192, conformer_blocks=3,
                    use_delta=True, use_delta2=True, use_se=True)
        epochs = 150
        kw = dict(BASE)
        if name == "roastformer":
            pass
        elif name == "roastformer_nodelta":      # no temporal-difference streams
            kw.update(use_delta=False, use_delta2=False)
        elif name == "roastformer_nodelta2":     # keep delta, drop delta-delta
            kw.update(use_delta2=False)
        elif name == "roastformer_nose":         # drop squeeze-excitation
            kw.update(use_se=False)
        elif name == "roastformer_noconf":       # drop conformer temporal blocks
            kw.update(conformer_blocks=0)
        elif name == "roastformer_nopool":       # mean pool instead of attentive stats
            kw.update(use_attn_pool=False)
        elif name == "roastformer_nofreq":       # mean pool instead of freq attention
            kw.update(use_freq_attn=False)
        # --- legacy tuning-search configs (kept for reproducibility) ---
        elif name == "roastformer_base":         # original 0.84M / 80ep design
            kw = dict(width=(40, 56, 80), d=128, conformer_blocks=2,
                      use_delta=True, use_delta2=False, use_se=False)
            epochs = 80
        elif name == "roastformer_v1":
            kw = dict(width=(48, 72, 96), d=160, conformer_blocks=3); epochs = 100
        elif name == "roastformer_v2":
            kw = dict(width=(40, 56, 80), d=128, conformer_blocks=2); epochs = 150
        elif name == "roastformer_v3":
            kw = dict(width=(48, 72, 96), d=160, conformer_blocks=3); epochs = 150
        elif name == "roastformer_v4":
            kw = dict(BASE); epochs = 150
        elif name in ("roastformer_v5", "roastformer_scaleonly"):
            kw = dict(width=(64, 96, 128), d=192, conformer_blocks=3); epochs = 150
        else:
            raise ValueError(f"unknown model {name}")
        return RoastFormer(num_classes, **kw), dict(input="mel32", epochs=epochs, batch_size=64,
                                                    lr=1e-3, weight_decay=1e-4, mixup=0.2, specaug=True)
    raise ValueError(f"unknown model {name}")


ALL_MODELS = ["cnn14", "conformer", "ast", "bcresnet", "passt", "htsat",
              "wavlm", "mobilenetv3", "audiomamba", "roastformer"]  # + 'clap' handled separately


def count_params(model):
    """Total model size (frozen weights included) so the comparison table
    uses one consistent metric across fine-tuned and from-scratch models."""
    return sum(p.numel() for p in model.parameters())
