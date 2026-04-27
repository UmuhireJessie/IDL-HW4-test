import torch.nn as nn
import torch
import random
from typing import Tuple, Optional, Literal
from .masks import PadMask, CausalMask
from .positional_encoding import PositionalEncoding
from .decoder_layers import SelfAttentionDecoderLayer, CrossAttentionDecoderLayer
from .encoder_layers import SelfAttentionEncoderLayer
from .speech_embedding import SpeechEmbedding
import warnings
from torchinfo import summary


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout, max_len, num_classes,
                 weight_tying=False, layer_drop_rate=0.0):
        super().__init__()
        self.max_len = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.dec_layers = nn.ModuleList([
            SelfAttentionDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.target_embedding = nn.Embedding(num_classes, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.final_linear = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight

    def forward(self, padded_targets, target_lengths=None):
        if self.training and target_lengths is None:
            raise ValueError("target_lengths must be provided during training")

        pad_mask_dec = None
        if target_lengths is not None:
            pad_mask_dec = PadMask(padded_targets.unsqueeze(-1), target_lengths)

        causal_mask = CausalMask(padded_targets.unsqueeze(-1))

        x = self.dropout(self.positional_encoding(self.target_embedding(padded_targets)))

        running_att = {}
        for i in range(self.num_layers):
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue
            x, attention = self.dec_layers[i](x, key_padding_mask=pad_mask_dec, attn_mask=causal_mask)
            running_att[f'layer{i+1}_dec_self'] = attention

        seq_out = self.final_linear(self.norm(x))
        return seq_out, running_att

    def score(self, batch_prompts):
        if self.training:
            raise ValueError("score method is not supported during training")
        seq_out, _ = self.forward(batch_prompts, target_lengths=None)
        return seq_out[:, -1, :]


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, input_dim, time_reduction, reduction_method, num_encoder_layers,
                 num_encoder_heads, d_ff_encoder, num_decoder_layers, num_decoder_heads,
                 d_ff_decoder, d_model, dropout, max_len, num_classes,
                 weight_tying=False, layer_drop_rate=0.0,
                 skip_encoder_pe=False, skip_decoder_pe=False):
        super().__init__()
        self.max_len = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes = num_classes
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.skip_encoder_pe = skip_encoder_pe
        self.skip_decoder_pe = skip_decoder_pe

        self.enc_layers = nn.ModuleList([
            SelfAttentionEncoderLayer(d_model, num_encoder_heads, d_ff_encoder, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.dec_layers = nn.ModuleList([
            CrossAttentionDecoderLayer(d_model, num_decoder_heads, d_ff_decoder, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.source_embedding = SpeechEmbedding(input_dim, d_model, time_reduction, reduction_method, dropout)
        self.target_embedding = nn.Embedding(num_classes, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.final_linear = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

        self.ctc_head = nn.Sequential(
            nn.Linear(d_model, num_classes),
            nn.LogSoftmax(dim=-1)
        )

        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight

    def encode(self, padded_sources, source_lengths):
        x_enc, x_enc_lengths = self.source_embedding(padded_sources, source_lengths)

        if not self.skip_encoder_pe:
            x_enc = self.positional_encoding(x_enc)
        x_enc = self.dropout(x_enc)

        pad_mask_src = PadMask(x_enc, x_enc_lengths.long())

        running_att = {}
        for i in range(self.num_encoder_layers):
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue
            x_enc, attention = self.enc_layers[i](x_enc, key_padding_mask=pad_mask_src)
            running_att[f'layer{i+1}_enc_self'] = attention

        x_enc = self.encoder_norm(x_enc)
        ctc_logits = self.ctc_head(x_enc)  # (B, T, num_classes)

        ctc_inputs = {
            'log_probs': ctc_logits.permute(1, 0, 2),  # (T, B, num_classes)
            'lengths': x_enc_lengths.long()
        }
        return x_enc, pad_mask_src, running_att, ctc_inputs

    def decode(self, padded_targets, encoder_output, target_lengths=None, pad_mask_src=None):
        pad_mask_tgt = None
        if target_lengths is not None:
            pad_mask_tgt = PadMask(padded_targets.unsqueeze(-1), target_lengths)

        if pad_mask_tgt is None and self.training:
            warnings.warn("pad_mask_tgt is None")

        causal_mask = CausalMask(padded_targets.unsqueeze(-1))

        x_dec = self.target_embedding(padded_targets)
        if not self.skip_decoder_pe:
            x_dec = self.positional_encoding(x_dec)
        x_dec = self.dropout(x_dec)

        running_att = {}
        for i in range(self.num_decoder_layers):
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue
            x_dec, self_attn, cross_attn = self.dec_layers[i](
                x_dec, encoder_output,
                dec_key_padding_mask=pad_mask_tgt,
                enc_key_padding_mask=pad_mask_src,
                attn_mask=causal_mask
            )
            running_att[f'layer{i+1}_dec_self'] = self_attn
            running_att[f'layer{i+1}_dec_cross'] = cross_attn

        seq_out = self.final_linear(self.decoder_norm(x_dec))
        return seq_out, running_att

    def forward(self, padded_sources, padded_targets, source_lengths=None, target_lengths=None):
        if self.training and target_lengths is None:
            raise ValueError("target_lengths must be provided during training")
        if self.training and source_lengths is None:
            raise ValueError("source_lengths must be provided during training")

        encoder_output, pad_mask_src, enc_running_att, ctc_inputs = self.encode(padded_sources, source_lengths)
        seq_out, dec_running_att = self.decode(padded_targets, encoder_output, target_lengths, pad_mask_src)

        running_att = {**enc_running_att, **dec_running_att}
        return seq_out, running_att, ctc_inputs

    def score(self, batch_prompts, encoder_output, pad_mask_src):
        if self.training:
            raise ValueError("score method is not supported during training")
        seq_out, _ = self.decode(batch_prompts, encoder_output, None, pad_mask_src)
        return seq_out[:, -1, :]

    @classmethod
    def from_pretrained_decoder(cls, decoder_checkpoint_path, config):
        print("\n=== Initializing Encoder-Decoder from Pretrained Decoder ===")
        print(f"Loading checkpoint from: {decoder_checkpoint_path}")
        print("\nCreating new encoder-decoder model...")
        model = cls(**config)

        print("Loading pretrained decoder weights...")
        checkpoint = torch.load(decoder_checkpoint_path, map_location='cpu', weights_only=True)
        decoder_state_dict = checkpoint['model_state_dict']

        transferred_params = []
        new_params = []

        def transfer_module_weights(target_module, prefix):
            module_state_dict = {
                k.replace(prefix, ''): v
                for k, v in decoder_state_dict.items()
                if k.startswith(prefix)
            }
            param_count = sum(p.numel() for p in target_module.parameters())
            print(f"  - Transferring {prefix} ({param_count:,} parameters)")
            target_module.load_state_dict(module_state_dict)
            for name, param in target_module.named_parameters():
                transferred_params.append((f"{prefix}{name}", param))

        print("\nTransferring shared components:")
        transfer_module_weights(model.target_embedding, 'target_embedding.')
        transfer_module_weights(model.final_linear, 'final_linear.')
        transfer_module_weights(model.decoder_norm, 'norm.')

        num_layers = min(
            len([k for k in decoder_state_dict.keys() if k.startswith('dec_layers.')]) // 2,
            model.num_decoder_layers
        )
        print(f"\nTransferring decoder layers (found {num_layers} layers):")
        for i in range(num_layers):
            print(f"\nLayer {i + 1}/{num_layers}:")
            transfer_module_weights(model.dec_layers[i].self_attn, f'dec_layers.{i}.self_attn.')
            transfer_module_weights(model.dec_layers[i].ffn, f'dec_layers.{i}.ffn.')

        print("\nCollecting new parameters...")
        for name, param in model.named_parameters():
            is_new = True
            for transferred_name, transferred_param in transferred_params:
                if param is transferred_param:
                    is_new = False
                    break
            if is_new:
                new_params.append((name, param))

        print("\n=== Initialization Complete ===")
        return model, {'transferred': transferred_params, 'new': new_params}

    def log_param_groups(self, param_groups):
        print("\nParameter groups:")
        total_params = 0
        total_trainable = 0
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            trainable = sum(p.numel() for p in group['params'] if p.requires_grad)
            total_params += num_params
            total_trainable += trainable
            print(f"\n{group['name']}:")
            print(f"  Parameters: {num_params:,}")
            print(f"  Trainable: {trainable:,}")
            print(f"  LR factor: {group['lr_factor']}")
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Total trainable: {total_trainable:,}")


def get_decoder_only_inputs(max_len=300, num_classes=10000):
    batch_size = 8
    padded_targets = torch.randint(0, num_classes, (batch_size, max_len))
    source_lengths = torch.ones(batch_size) * max_len
    return padded_targets, source_lengths

def get_encoder_decoder_inputs(max_len=300, num_classes=10000):
    batch_size = 8
    padded_targets = torch.randint(0, num_classes, (batch_size, max_len))
    source_lengths = torch.ones(batch_size) * max_len
    return padded_targets, source_lengths

def test_decoder_only(num_layers=12, num_heads=8, d_model=512, d_ff=2048, dropout=0.1, max_len=300, num_classes=1000):
    padded_targets, target_lengths = get_decoder_only_inputs(max_len, num_classes)
    model = DecoderOnlyTransformer(num_layers, d_model, num_heads, d_ff, dropout, max_len, num_classes)
    summary(model, input_data=[padded_targets, target_lengths])

if __name__ == "__main__":
    test_decoder_only()
