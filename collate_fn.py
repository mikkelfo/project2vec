""" Implements all collate functionality """
 
import bisect
from typing import List, Dict, Tuple, Union
import torch
from utils import get_max_special_token_value, mask_inputs
 

class Collate:
    """Implements the core collate functions"""
 
    def __init__(self, truncate_length=512, background_length=0, segment=False):
        self.truncate_length = truncate_length
        self.background_length = background_length
        self.max_seq_len = truncate_length + background_length
        self.bg_events = int(self.background_length > 0)
        self.segment = segment

        self.len_buckets = [
            128 * (2**i) for i in range((self.max_seq_len // 128).bit_length())
        ]
        if self.max_seq_len not in self.len_buckets:
            self.len_buckets.append(self.max_seq_len)
 
    def __call__(self, batch: List[dict]) -> dict:
        data_keys = [key for key in batch[0]] + ["segment"] * int(self.segment)
        output = {key: [] for key in data_keys + ["sequence_lens"]}

        for person in batch:
            indv = self.process_person(person)
            for key, v in indv.items():
                output.setdefault(key, []).append(v)  # Allows arbitrary keys in indv

            # Add padding information
            output["sequence_lens"].append(len(indv["event"]))

        # Pad all person keys
        for key in data_keys:
            output[key] = self._pad(output[key])
        output["sequence_lens"] = torch.as_tensor(output["sequence_lens"])

        output["padding_mask"] = output["event"] == 0

        return output

    def process_person(self, person, extra_info=False):
        """Handles all person-specific processsing"""
        output = {}
        # Start with events
        person_seq, person_event_lens, event_border = self._flatten(person.pop("event"))
        output["event"] = person_seq

        if self.segment:
            person["segment"] = list(range(1, len(person_event_lens) + 1))
        # Add rest of keys
        for key in person:
            truncated_seq = self._truncate(person[key], event_border)
            expanded_seq = self.expand(truncated_seq, person_event_lens)
            output[key] = expanded_seq

        if extra_info:  # Only needed for other collates or debugging
            output["event_lens"] = person_event_lens
        return output

    def _truncate(self, seq, event_border: int):
        if event_border is None:
            return seq
        return seq[: self.bg_events] + seq[event_border:]

    @staticmethod
    def expand(seq: list, repeats: list) -> torch.Tensor:
        """Repeats seq[i] repeats[i] times"""
        return torch.repeat_interleave(torch.as_tensor(seq), torch.as_tensor(repeats))

    def _flatten(self, events: List[List[int]]):
        """Flattens events and (optional) truncates, returning flatten_seq and the last event idx"""
        person_seq, person_event_lens, event_border = (
            self._flatten_reverse_and_truncate(events, self.truncate_length)
        )
        return person_seq, person_event_lens, event_border

    def _flatten_reverse_and_truncate(
        self, sequence: List, truncate_length: int
    ) -> Tuple[list, list, int]:
        """Flattens a reversed list (keeping newest info) until truncate_length reached, adds background and then returns event_idx (if terminated) and list"""
        result, event_lens = [], []
        total_length = 0
        for i, sublist in enumerate(reversed(sequence)):
            n = len(sublist)
            total_length += n
            if total_length > truncate_length:
                break
            event_lens.append(n)
            result.extend(sublist[::-1])
        else:  # If loop finished (total_length < truncate_length)
            return result[::-1], event_lens[::-1], None

        # Add background onto it
        for sublist in reversed(sequence[: self.bg_events]):
            result.extend(sublist[::-1])
            event_lens.append(len(sublist))
        return result[::-1], event_lens[::-1], -i

    def _pad(
        self,
        sequence: Union[list, torch.Tensor],
        dtype: torch.dtype = None,
        padding_value=0,
    ) -> torch.Tensor:
        """Pads the sequence (using padding_value) to closest bucket in self.len_buckets and converts to tensor"""
        # Conver to tensors and get max_len and max_idx
        sequences = [torch.as_tensor(seq, dtype=dtype) for seq in sequence]
        max_len, max_idx = torch.max(
            torch.tensor([s.size(0) for s in sequences]), dim=0
        )

        # If batch does not match a predefined len bucket
        if max_len.values not in self.len_buckets:
            closest_len = self.len_buckets[
                bisect.bisect_left(self.len_buckets, max_len)
            ]
            extra_dims = sequences[max_idx].shape[1:]
            sequences[max_idx] = torch.cat(
                (
                    sequences[max_idx],
                    torch.full((closest_len - max_len, *extra_dims), padding_value),
                )
            )
        return torch.nn.utils.rnn.pad_sequence(
            sequences,
            batch_first=True,
            padding_value=padding_value,
        )


class MaskCollate(Collate):
    """Standard collate with masking"""

    def __init__(
        self,
        vocab: dict,
        mask_prob=0.15,
        replace_prob=0.8,
        random_prob=0.1,
        truncate_length=512,
        background_length=0,
        segment=False,
    ):
        super().__init__(truncate_length, background_length, segment=segment)
        self.vocab = vocab
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_prob = random_prob
        self.special_token_border = get_max_special_token_value(vocab)

    def __call__(self, batch: List[Dict]) -> Dict:
        batch = super().__call__(batch)

        # Mask events and produce targets
        batch["event"], batch["target"] = mask_inputs(
            batch["event"],
            self.vocab,
            mask_prob=self.mask_prob,
            replace_prob=self.replace_prob,
            random_prob=self.random_prob,
            special_token_border=self.special_token_border,
        )
        return batch


class CensorCollate(Collate):
    """Standard collate with censoring"""

    def __init__(
        self,
        truncate_length: int,
        background_length: int,
        segment: bool,
        negative_censor: int = 0,
    ):
        super().__init__(truncate_length, background_length, segment=segment)
        self.negative_censor = negative_censor

    def __call__(self, batch: List[Dict]) -> Dict:
        """Input: List of Batch, Outcomes"""
        targets = [person.pop("target") for person in batch]
        data = self._censor(batch)

        batch = super().__call__(data)
        batch["targets"] = self._adjust_targets(targets)
        return batch

    def _censor(self, data: List[Dict]) -> List[Dict]:
        censored_data = []
        for person in data:
            censored_person = self._censor_person(person, None) # TODO: Need censoring date
            censored_data.append(censored_person)
        return censored_data

    def _censor_person(self, person, censor_abspos):
        abspos = person["abspos"]
        if censor_abspos is None:
            censor_abspos = (
                abspos[-1] + 1e-10  # Tiny float adjust if self.negative_censor=0
            ) - self.negative_censor
        # Since data is sorted by abspos, we can just do binary search
        last_valid_idx = bisect.bisect_left(abspos, censor_abspos)
        last_valid_idx = max(last_valid_idx, self.bg_events)  # Always keep background
        return {key: value[:last_valid_idx] for key, value in person.items()}

    def _adjust_targets(self, targets):
        return torch.as_tensor(targets, dtype=torch.float32).unsqueeze(1)
