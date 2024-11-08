""" Implements all collate functionality """
 
import bisect
import itertools
from typing import List, Dict, Any, Tuple, Optional
import torch
from utils import get_max_special_token_value, mask_inputs
 

class Collate:
    """Implements different collate functions"""
 
    def __init__(self, truncate_length=512, background_length=0, cls_token=False):
        self.truncate_length = truncate_length
        self.background_length = background_length
        self.max_seq_len = truncate_length + background_length
        self.first_event = 1 + int(cls_token)
 
    def __call__(self, batch: List[Dict]):
        """For equal number of tokens and flatten"""
        output = {}
        keys = [key for key in batch[0] if key != "event"]
        # First truncate and flatten events
        events = self._collect_values_by_key(batch, "event")
        flatten_events, event_lens, event_borders = [], [], []
        for person_events in events:
            person_event_lens = [len(e) for e in person_events]
            # If person has more tokens than allowed, truncate (from behind) and flatten to keep newest info
            if sum(person_event_lens) > self.max_seq_len:
                event_border, person_seq = self._flatten_reverse(
                    person_events, self.truncate_length
                )
                # Truncate event_lens
                if self.background_length > 0:  # Add background & CLS # TODO: Slow O(n)
                    person_event_lens_bg = person_event_lens[: self.first_event]
                    person_seq_bg = sum(person_events[: self.first_event], [])
                else:
                    person_event_lens_bg, person_seq_bg = [], []
 
                person_event_lens = (
                    person_event_lens_bg + person_event_lens[event_border:]
                )
                person_seq = person_seq_bg + person_seq
            else:  # No truncation, do as normal
                event_border = None
                person_seq = list(itertools.chain.from_iterable(person_events))
            # Append relevant outputs
            flatten_events.append(torch.as_tensor(person_seq, dtype=torch.int32))
            event_lens.append(person_event_lens)
            event_borders.append(event_border)
        output["event"] = self._pad(flatten_events, dtype=torch.int32, padding_value=0)
        output["last_data_idx"] = torch.tensor(
            [len(seq) for seq in flatten_events], dtype=torch.int16
        )
        # TODO: Dont save these (only saved for CausalEventCollate) - saved as list so it doens't take up GPU memory and time
        output["event_lens"] = event_lens
        output["event_borders"] = event_borders
 
        # Add rest of keys
        for key in keys:
            sequences = self._collect_values_by_key(batch, key)
            result = []
            for i, seq in enumerate(sequences):
                # Add background if truncation has happened (event_border None), else skip
                truncated_seq = self._truncate(seq, event_borders[i])
                expanded_seq = torch.repeat_interleave(
                    truncated_seq, torch.as_tensor(event_lens[i])
                )
                result.append(expanded_seq)
            output[key] = self._pad(result, padding_value=0)
 
        return output
 
    def _truncate(self, seq, event_border: int):
        background = [] if event_border is None else seq[: self.first_event]
        return torch.as_tensor(
            background + seq[event_border:],
        )
 
    @staticmethod
    def _collect_values_by_key(batch: List[Dict], key: str) -> List[List]:
        return [item[key] for item in batch]
 
    @staticmethod
    def _flatten_reverse(sequence: List, truncate_length: int) -> Tuple[int, list]:
        """Flattens a reversed list (keeping newest info) until truncate_length reached, then returns event_idx (if terminated) and list"""
        result = []
        total_length = 0
        for i, sublist in enumerate(reversed(sequence)):
            total_length += len(sublist)
            if total_length > truncate_length:
                break
            result.extend(sublist[::-1])
        return -i, result[::-1]
 
    def _pad(
        self, sequence: Any, dtype: Optional[torch.dtype] = None, padding_value=0
    ) -> torch.Tensor:
        """Pads the sequence (using padding_value) and converts to tensor"""
        sequences = [torch.as_tensor(seq, dtype=dtype) for seq in sequence]
        first_len = sequences[0].size(0)
        # lens = torch.tensor([seq.size(0) for seq in sequences])
        # max_len, max_i = torch.max(lens, dim=0)
        if first_len != self.max_seq_len:
            # if (r := max_len % 128) != 0:
            add_dims = sequences[0].shape[1:]  # Makes it work with multiple dimensions
            sequences[0] = torch.cat(
                (
                    sequences[0],
                    # torch.full((128 - r, *add_dims), padding_value),
                    torch.full(
                        (self.max_seq_len - first_len, *add_dims), padding_value
                    ),
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
        cls_token=False,
    ):
        super().__init__(truncate_length, background_length, cls_token=cls_token)
        self.vocab = vocab
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_prob = random_prob
        self.special_token_border = get_max_special_token_value(vocab)
 
    def __call__(self, batch: List[Dict]) -> Dict:
        batch = super().__call__(batch)
        return self._mask(batch)
 
    def _mask(self, batch: Dict) -> Dict:
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
 
    def __call__(self, batch: List[Tuple[Dict]]) -> Dict:
        """Input: List of Batch, Outcomes"""
        data, outcome_info = zip(*batch, strict=True)
        data = self._censor(data, outcome_info)
 
        batch = super().__call__(data)
        batch = self._add_outcome_info(batch, outcome_info)
        return batch
 
    @staticmethod
    def _censor(data: Tuple[Dict], outcome_info: Tuple[Dict]) -> List[Dict]:
        censored_data = []
        for person, outcome in zip(data, outcome_info, strict=True):
            abspos = person["abspos"]
            censor_abspos = outcome["censor"]
            # Since data is sorted, we can just do binary search
            last_valid_idx = bisect.bisect_left(abspos, censor_abspos)
            valid_person = {
                key: value[:last_valid_idx] for key, value in person.items()
            }
            censored_data.append(valid_person)
        return censored_data
 
    @staticmethod
    def _add_outcome_info(batch: dict, outcome_info: List[dict]) -> Dict:
        batch.update(
            {key: [out[key] for out in outcome_info] for key in outcome_info[0]}
        )
        batch["target"] = torch.tensor(
            [out["target"] for out in outcome_info], dtype=torch.float32
        )
        return batch
 

class CausalEventCollate(Collate):
    """Collate with causal event mask and multi-trajectory targets"""
 
    def __init__(
        self,
        truncate_length: int,
        background_length: int,
        cls_token: bool,
        prediction_windows: list,
        negative_censor: int,
    ):
        super().__init__(truncate_length, background_length, cls_token)
        self.prediction_windows = torch.as_tensor(prediction_windows)
        self.negative_censor = negative_censor
 
    def __call__(self, batch: List[Tuple[Dict]]) -> Dict:
        data, outcome_info = zip(*batch, strict=True)
        data, targets = self._censor_and_create_targets(data, outcome_info)
        batch = super().__call__(data)
 
        # Get event_mask (risk trajectories) and event intervals (flex attention)
        event_intervals, event_mask = self._get_causal_event_mask_flex(
            batch,
            max_seq_len=batch["event"].size(1),
        )
        batch["event_mask"] = event_mask
        batch["event_intervals"] = event_intervals
        batch = self._add_outcome_info(batch, outcome_info, targets)
        return batch
 
    def _add_outcome_info(
        self, batch: dict, outcome_info: List[dict], targets: List
    ) -> Dict:
        batch.update(
            {key: [out[key] for out in outcome_info] for key in outcome_info[0]}
        )
        truncated_targets = []
        for i, t in enumerate(targets):
            truncated_targets.append(self._truncate(t, batch["event_borders"][i]))
        batch["target"] = self._pad(
            truncated_targets, dtype=torch.float32, padding_value=-100
        )
        return batch
 
    def _get_causal_event_mask_flex(self, batch, max_seq_len) -> torch.tensor:
        padded_event_lens = self._pad(
            [list(itertools.accumulate(seq)) for seq in batch["event_lens"]],
            dtype=torch.int32,
            padding_value=10_000,  # Set to higher number than max_seq_len
        )
 
        range_tensor = torch.arange(max_seq_len)
        event_mask = (range_tensor < padded_event_lens.unsqueeze(-1)) & (
            range_tensor < batch["last_data_idx"].unsqueeze(-1)
        ).unsqueeze(1)
        return padded_event_lens, event_mask.half()
 
    def _censor_and_create_targets(
        self, data: Tuple[Dict], outcome_info: Tuple[Dict]
    ) -> List[Dict]:
        censored_data = []
        targets = []
        for person, outcome in zip(data, outcome_info, strict=True):
            abspos = person["abspos"]
            censor_abspos = outcome["censor"]
            if censor_abspos is None:
                censor_abspos = abspos[-1] - self.negative_censor  # Adjust censoring
 
            # Since data is sorted, we can just do binary search
            last_valid_idx = bisect.bisect_left(abspos, censor_abspos)
            if (last_valid_idx == 0) and (self.background_length > 0):
                last_valid_idx = 1
            valid_person = {
                key: value[:last_valid_idx] for key, value in person.items()
            }
            # Create targets
            valid_targets = (
                torch.tensor(valid_person["abspos"]).unsqueeze(1)
                + self.prediction_windows
            ) > censor_abspos
            targets.append(valid_targets)
            censored_data.append(valid_person)
        return censored_data, targets
 
    def _truncate(self, seq, event_border: int):
        if event_border is None:
            return torch.as_tensor(seq)
        else:
            if isinstance(seq, list):  # TODO: check sequences[0] so it's just once
                return torch.as_tensor(seq[: self.first_event] + seq[event_border:])
            else:
                return torch.cat((seq[: self.first_event], seq[event_border:]))