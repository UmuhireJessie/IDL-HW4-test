import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer


class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.score_fn  = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device    = device

    def _apply_repeat_penalty(self, logits, sequences, penalty=1.0):
        if penalty == 1.0:
            return logits
        if logits.dim() == 2:
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0 / penalty)
                )
        else:
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    logits[batch_idx, beam_idx, unique_tokens] = logits[batch_idx, beam_idx, unique_tokens] / torch.where(
                        logits[batch_idx, beam_idx, unique_tokens] > 0,
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], penalty),
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], 1.0 / penalty)
                    )
        return logits

    def _filter_logits(self, logits, temperature=1.0, top_k=0, top_p=1.0):
        logits = logits / temperature
        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_k_logits[..., -1:]] = float('-inf')
        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        return logits

    def generate_greedy(self, x, temperature=1.0, repeat_penalty=1.0):
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")

        batch_size = x.size(0)
        scores     = torch.zeros(batch_size, device=x.device)
        finished   = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            if finished.all():
                break
            logits      = self.score_fn(x)
            logits      = self._apply_repeat_penalty(logits, x, repeat_penalty)
            logits      = logits / temperature
            log_probs   = torch.log_softmax(logits, dim=-1)
            next_tokens = log_probs.argmax(dim=-1)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
            scores      = torch.where(finished, scores, scores + token_scores)
            x           = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)
            finished    = finished | (next_tokens == self.tokenizer.eos_id)

        return x, scores

    def generate_beam(self, x, beam_width, temperature=1.0, repeat_penalty=1.0):
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")

        batch_size = x.size(0)
        vocab_size = self.tokenizer.vocab_size

        # ── Step 0: first forward pass on the prompt ──────────────────────
        # x: (B, seq)  →  score_fn always sees exactly (B, seq)
        logits    = self.score_fn(x)                            # (B, V)
        logits    = self._apply_repeat_penalty(logits, x, repeat_penalty)
        logits    = logits / temperature
        log_probs = torch.log_softmax(logits, dim=-1)           # (B, V)

        top_scores, top_tokens = torch.topk(log_probs, beam_width, dim=-1)  # (B, W)
        scores   = top_scores                                   # (B, W)
        finished = (top_tokens == self.tokenizer.eos_id)        # (B, W)

        # Expand: (B, seq) → (B, W, seq+1)
        # .contiguous() is critical — without it all beams share memory
        x = x.unsqueeze(1).expand(-1, beam_width, -1).contiguous()  # (B, W, seq)
        x = torch.cat([x, top_tokens.unsqueeze(-1)], dim=-1)         # (B, W, seq+1)

        # ── Subsequent steps ──────────────────────────────────────────────
        for _ in range(self.max_length - x.size(2)):
            if finished.all():
                break

            # Call score_fn once per beam, always with shape (B, seq)
            # This keeps batch_size=B so any batch-indexed score_fn works correctly
            beam_logits = []
            for w in range(beam_width):
                lw = self.score_fn(x[:, w, :].contiguous())    # (B, V)
                beam_logits.append(lw)
            # Stack → (B, W, V)
            next_scores = torch.stack(beam_logits, dim=1)
            next_scores = self._apply_repeat_penalty(next_scores, x, repeat_penalty)
            next_scores = next_scores / temperature
            next_scores = torch.log_softmax(next_scores, dim=-1)  # (B, W, V)

            # Cumulative log-probs
            cum_scores = scores.unsqueeze(-1) + next_scores     # (B, W, V)

            # Finished beams: block all expansions, allow EOS to keep score
            cum_scores[finished] = float('-inf')
            # vectorised: for each finished (b,w), set EOS slot = frozen score
            cum_scores[finished, self.tokenizer.eos_id] = scores[finished]

            # Select top-W globally
            flat          = cum_scores.view(batch_size, -1)     # (B, W*V)
            top_cum, top_idx = torch.topk(flat, beam_width, dim=-1)

            src_beams   = top_idx // vocab_size                 # (B, W)
            next_tokens = top_idx %  vocab_size                 # (B, W)
            scores      = top_cum

            # Reorder sequences and finished flags
            batch_idx = torch.arange(batch_size, device=x.device).unsqueeze(1)
            x        = x[batch_idx, src_beams].contiguous()     # (B, W, seq)
            finished = finished[batch_idx, src_beams]

            # Append next tokens
            x        = torch.cat([x, next_tokens.unsqueeze(-1)], dim=-1)
            finished = finished | (next_tokens == self.tokenizer.eos_id)

        # Sort beams best → worst
        sorted_scores, sort_idx = scores.sort(dim=-1, descending=True)
        batch_idx = torch.arange(batch_size, device=x.device).unsqueeze(1)
        x         = x[batch_idx, sort_idx].contiguous()
        scores    = sorted_scores

        return x, scores

    def generate_sample(self, x, temperature=1.0, top_k=0, top_p=1.0):
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")

        batch_size = x.size(0)
        scores     = torch.zeros(batch_size, device=x.device)
        finished   = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            if finished.all():
                break
            next_scores     = self.score_fn(x)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs       = torch.log_softmax(filtered_logits, dim=-1)
            probs           = torch.exp(log_probs)
            next_tokens     = torch.multinomial(probs, num_samples=1).squeeze(-1)
            token_scores    = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
            scores          = torch.where(finished, scores, scores + token_scores)
            x               = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)
            finished        = finished | (next_tokens == self.tokenizer.eos_id)

        return x, scores

    @staticmethod
    def post_process_sequence(seq, tokenizer):
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                return seq[:eos_indices[0].item() + 1]
            return seq
        eos_mask    = seq == tokenizer.eos_id
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        seq_mask    = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]
