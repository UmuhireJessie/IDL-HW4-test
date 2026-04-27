from typing import Literal, Tuple, Optional
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as tat
from .tokenizer import H4Tokenizer


class ASRDataset(Dataset):
    def __init__(self, partition, config, tokenizer, isTrainPartition, global_stats=None):
        self.config           = config
        self.partition        = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer        = tokenizer

        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        # Full paths for loading
        self.fbank_dir   = os.path.join(config['root'], partition, 'fbank')
        self.fbank_files = sorted([
            os.path.join(self.fbank_dir, f)
            for f in os.listdir(self.fbank_dir)
            if f.endswith('.npy')
        ])

        subset_size      = int(len(self.fbank_files) * config.get('subset', 1.0))
        self.fbank_files = self.fbank_files[:subset_size]
        self.length      = len(self.fbank_files)

        if self.partition != "test-clean":
            self.text_dir   = os.path.join(config['root'], partition, 'text')
            self.text_files = sorted([
                os.path.join(self.text_dir, f)
                for f in os.listdir(self.text_dir)
                if f.endswith('.npy')
            ])
            self.text_files = self.text_files[:subset_size]

            if len(self.fbank_files) != len(self.text_files):
                raise ValueError("Number of feature and transcript files must match")

            # Verify alignment by basename
            for fbank_file, text_file in zip(self.fbank_files, self.text_files):
                if os.path.basename(fbank_file) != os.path.basename(text_file):
                    raise ValueError(
                        f"FBANK file {fbank_file} and TRANSCRIPT file {text_file} are misaligned."
                    )

        self.feats                 = []
        self.transcripts_shifted   = []
        self.transcripts_golden    = []
        self.total_chars           = 0
        self.total_tokens          = 0
        self.feat_max_len          = 0
        self.text_max_len          = 0

        if self.config['norm'] == 'global_mvn' and global_stats is None:
            if not isTrainPartition:
                raise ValueError("global_stats must be provided for non-training partitions when using global_mvn")
            count = 0
            mean  = torch.zeros(self.config['num_feats'], dtype=torch.float64)
            M2    = torch.zeros(self.config['num_feats'], dtype=torch.float64)

        print(f"Loading data for {partition} partition...")
        for i in tqdm(range(self.length)):
            # Load features — shape (num_feats, time)
            feat = np.load(self.fbank_files[i])
            feat = feat[:self.config['num_feats'], :]
            self.feats.append(feat)
            self.feat_max_len = max(self.feat_max_len, feat.shape[1])

            if self.config['norm'] == 'global_mvn' and global_stats is None:
                feat_tensor = torch.FloatTensor(feat)
                batch_count = feat_tensor.shape[1]
                count      += batch_count
                delta       = feat_tensor - mean.unsqueeze(1)
                mean       += delta.mean(dim=1)
                delta2      = feat_tensor - mean.unsqueeze(1)
                M2         += (delta * delta2).sum(dim=1)

            if self.partition != "test-clean":
                # Robust transcript loading — handles scalar and array formats
                loaded = np.load(self.text_files[i], allow_pickle=True)
                if loaded.ndim == 0:
                    transcript = str(loaded.item())
                else:
                    transcript = ''.join(loaded) if isinstance(loaded[0], str) else str(loaded[0])

                self.total_chars  += len(transcript)
                tokenized          = tokenizer.encode(transcript)
                self.total_tokens += len(tokenized)
                self.text_max_len  = max(self.text_max_len, len(tokenized) + 1)

                # Store as plain lists — decode() expects a list, not a tensor
                self.transcripts_shifted.append([self.sos_token] + tokenized)
                self.transcripts_golden.append(tokenized + [self.eos_token])

        self.avg_chars_per_token = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0

        if self.partition != "test-clean":
            if not (len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)):
                raise ValueError("Features and transcripts are misaligned")

        if self.config['norm'] == 'global_mvn':
            if global_stats is not None:
                self.global_mean, self.global_std = global_stats
            else:
                variance         = M2 / (count - 1)
                self.global_std  = torch.sqrt(variance + 1e-8).float()
                self.global_mean = mean.float()

        self.time_mask = tat.TimeMasking(
            time_mask_param=config['specaug_conf']['time_mask_width_range'],
            iid_masks=True
        )
        self.freq_mask = tat.FrequencyMasking(
            freq_mask_param=config['specaug_conf']['freq_mask_width_range'],
            iid_masks=True
        )

        # Overwrite with basenames so tests can compare fbank vs text filenames
        self.fbank_files = [os.path.basename(f) for f in self.fbank_files]
        if self.partition != "test-clean":
            self.text_files = [os.path.basename(f) for f in self.text_files]

    def get_avg_chars_per_token(self):
        return self.avg_chars_per_token

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        feat = torch.FloatTensor(self.feats[idx])   # (num_feats, time)

        if self.config['norm'] == 'global_mvn':
            feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)
        elif self.config['norm'] == 'cepstral':
            feat = (feat - feat.mean(dim=1, keepdim=True)) / (feat.std(dim=1, keepdim=True) + 1e-8)

        shifted_transcript, golden_transcript = None, None
        if self.partition != "test-clean":
            # Convert to tensor here — kept as list in storage so tokenizer.decode() works
            shifted_transcript = torch.LongTensor(self.transcripts_shifted[idx])
            golden_transcript  = torch.LongTensor(self.transcripts_golden[idx])

        return feat, shifted_transcript, golden_transcript

    def collate_fn(self, batch):
        batch_feats  = [feat.transpose(0, 1) for feat, _, _ in batch]  # (time, num_feats)
        feat_lengths = torch.LongTensor([f.shape[0] for f in batch_feats])
        padded_feats = pad_sequence(batch_feats, batch_first=True, padding_value=0.0)  # (B, T, F)

        padded_shifted, padded_golden, transcript_lengths = None, None, None
        if self.partition != "test-clean":
            batch_shifted      = [s for _, s, _ in batch]
            batch_golden       = [g for _, _, g in batch]
            transcript_lengths = torch.LongTensor([len(s) for s in batch_shifted])
            padded_shifted     = pad_sequence(batch_shifted, batch_first=True, padding_value=self.pad_token)
            padded_golden      = pad_sequence(batch_golden,  batch_first=True, padding_value=self.pad_token)

        if self.config["specaug"] and self.isTrainPartition:
            padded_feats = padded_feats.permute(0, 2, 1)   # (B, F, T)
            if self.config["specaug_conf"]["apply_freq_mask"]:
                for _ in range(self.config["specaug_conf"]["num_freq_mask"]):
                    padded_feats = self.freq_mask(padded_feats)
            if self.config["specaug_conf"]["apply_time_mask"]:
                for _ in range(self.config["specaug_conf"]["num_time_mask"]):
                    padded_feats = self.time_mask(padded_feats)
            padded_feats = padded_feats.permute(0, 2, 1)   # (B, T, F)

        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths
