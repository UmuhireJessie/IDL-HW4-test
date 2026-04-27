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
        # Store basic configuration
        self.config = config
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer

        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        root = config['root']
        self.fbank_dir = os.path.join(root, partition, 'fbank')
        self.fbank_files = sorted(os.listdir(self.fbank_dir))

        subset_size = int(len(self.fbank_files) * config.get('subset', 1.0))
        self.fbank_files = self.fbank_files[:subset_size]
        self.length = len(self.fbank_files)

        if self.partition != "test-clean":
            self.text_dir = os.path.join(root, partition, 'text')
            self.text_files = sorted(os.listdir(self.text_dir))
            self.text_files = self.text_files[:subset_size]
            if len(self.fbank_files) != len(self.text_files):
                raise ValueError("Number of feature and transcript files must match")

        self.feats = []
        self.transcripts_shifted = []
        self.transcripts_golden = []

        self.total_chars = 0
        self.total_tokens = 0
        self.feat_max_len = 0
        self.text_max_len = 0

        if self.config['norm'] == 'global_mvn' and global_stats is None:
            if not isTrainPartition:
                raise ValueError("global_stats must be provided for non-training partitions when using global_mvn")
            count = 0
            mean = torch.zeros(self.config['num_feats'], dtype=torch.float64)
            M2 = torch.zeros(self.config['num_feats'], dtype=torch.float64)

        print(f"Loading data for {partition} partition...")
        for i in tqdm(range(self.length)):
            feat = np.load(os.path.join(self.fbank_dir, self.fbank_files[i]))
            # feat shape: (num_feats, time) or (time, num_feats) - normalize to (num_feats, time)
            if feat.ndim == 2 and feat.shape[0] != self.config['num_feats']:
                feat = feat.T  # transpose to (num_feats, time)
            feat = feat[:self.config['num_feats'], :]  # truncate to num_feats

            self.feats.append(feat)
            self.feat_max_len = max(self.feat_max_len, feat.shape[1])

            if self.config['norm'] == 'global_mvn' and global_stats is None:
                feat_tensor = torch.FloatTensor(feat)
                batch_count = feat_tensor.shape[1]
                count += batch_count
                delta = feat_tensor - mean.unsqueeze(1)
                mean += delta.mean(dim=1)
                delta2 = feat_tensor - mean.unsqueeze(1)
                M2 += (delta * delta2).sum(dim=1)

            if self.partition != "test-clean":
                transcript = np.load(os.path.join(self.text_dir, self.text_files[i]))
                # transcript is a numpy array containing a string or array of strings
                if isinstance(transcript, np.ndarray):
                    transcript = str(transcript)
                    # clean up numpy string representation
                    transcript = transcript.strip()
                    if transcript.startswith("['") or transcript.startswith('["'):
                        transcript = transcript[2:-2]
                self.total_chars += len(transcript)

                tokenized = tokenizer.encode(transcript)
                self.total_tokens += len(tokenized)
                self.text_max_len = max(self.text_max_len, len(tokenized) + 1)

                shifted = torch.LongTensor([self.sos_token] + tokenized)
                golden = torch.LongTensor(tokenized + [self.eos_token])
                self.transcripts_shifted.append(shifted)
                self.transcripts_golden.append(golden)

        self.avg_chars_per_token = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0

        if self.partition != "test-clean":
            if not (len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)):
                raise ValueError("Features and transcripts are misaligned")

        if self.config['norm'] == 'global_mvn':
            if global_stats is not None:
                self.global_mean, self.global_std = global_stats
            else:
                variance = M2 / (count - 1)
                self.global_std = torch.sqrt(variance + 1e-8).float()
                self.global_mean = mean.float()

        self.time_mask = tat.TimeMasking(
            time_mask_param=config['specaug_conf']['time_mask_width_range'],
            iid_masks=True
        )
        self.freq_mask = tat.FrequencyMasking(
            freq_mask_param=config['specaug_conf']['freq_mask_width_range'],
            iid_masks=True
        )

    def get_avg_chars_per_token(self):
        return self.avg_chars_per_token

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        feat = torch.FloatTensor(self.feats[idx])  # (num_feats, time)

        if self.config['norm'] == 'global_mvn':
            feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)
        elif self.config['norm'] == 'cepstral':
            feat = (feat - feat.mean(dim=1, keepdim=True)) / (feat.std(dim=1, keepdim=True) + 1e-8)

        shifted_transcript, golden_transcript = None, None
        if self.partition != "test-clean":
            shifted_transcript = self.transcripts_shifted[idx]
            golden_transcript = self.transcripts_golden[idx]

        return feat, shifted_transcript, golden_transcript

    def collate_fn(self, batch):
        # batch is list of (feat, shifted, golden) tuples
        # feat: (num_feats, time) -> transpose to (time, num_feats) for padding
        batch_feats = [item[0].T for item in batch]  # list of (time, num_feats)
        feat_lengths = torch.LongTensor([f.shape[0] for f in batch_feats])
        padded_feats = pad_sequence(batch_feats, batch_first=True, padding_value=0.0)  # (B, T, F)

        padded_shifted, padded_golden, transcript_lengths = None, None, None
        if self.partition != "test-clean":
            batch_shifted = [item[1] for item in batch]
            batch_golden = [item[2] for item in batch]
            transcript_lengths = torch.LongTensor([s.shape[0] for s in batch_shifted])
            padded_shifted = pad_sequence(batch_shifted, batch_first=True, padding_value=self.pad_token)
            padded_golden = pad_sequence(batch_golden, batch_first=True, padding_value=self.pad_token)

        if self.config["specaug"] and self.isTrainPartition:
            padded_feats = padded_feats.permute(0, 2, 1)  # (B, F, T)
            if self.config["specaug_conf"]["apply_freq_mask"]:
                for _ in range(self.config["specaug_conf"]["num_freq_mask"]):
                    padded_feats = self.freq_mask(padded_feats)
            if self.config["specaug_conf"]["apply_time_mask"]:
                for _ in range(self.config["specaug_conf"]["num_time_mask"]):
                    padded_feats = self.time_mask(padded_feats)
            padded_feats = padded_feats.permute(0, 2, 1)  # (B, T, F)

        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths
