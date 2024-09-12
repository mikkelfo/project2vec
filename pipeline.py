""" File for tokenization and event creation """
 
import json
from typing import List
from pathlib import Path
import polars as pl
 
from tokenizer import create_vocab # type: ignore
from utils import get_pnrs
from features import (
    add_cls_token,
    add_sep_tokens,
    create_background,
    create_abspos,
    create_age,
    create_segment,
    create_tokenized_events,
)
 

class DataPipeline:
    """Class for handling everything related to data processing of the Datamodule"""
 
    def __init__(
        self,
        background: pl.DataFrame,
        cls_token: bool,
        sep_token: bool,
        segment: bool,
    ):
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.segment = segment
        assert {"person_id", "birthday"}.issubset(
            background.columns
        ), "Required cols: person_id, birthday"
        self.background = background
 
        # Assigned during __call__
        self.dir_path = None
 
    def __call__(self, sources: List[pl.LazyFrame], dir_path: Path = None):
        """Does all data processing required to create features"""
        self.dir_path = dir_path
 
        # Subset background to sources
        background_subset = self.get_background_subset(sources, self.background)
 
        # Get birthdates
        birthdates = self.get_lazy_birthdates(background_subset)
 
        # Get background and prepend to sources
        if not (set(background_subset.columns) == {"person_id", "birthday"}):
            background = self.get_lazy_background(background_subset)
            sources = [background] + sources
 
        # Get vocab if not computed
        vocab = self.get_vocab(sources)
 
        # Get tokenized event LataFrame
        tokenized_event_lf = self.get_lazy_tokenized_event_lf(sources, vocab)
 
        if self.cls_token:
            tokenized_event_lf = add_cls_token(
                lf=tokenized_event_lf,
                birthdates=birthdates,
                cls_token=vocab["[CLS]"],
            )
 
        if self.sep_token:
            tokenized_event_lf = add_sep_tokens(
                lf=tokenized_event_lf, sep_token=vocab["[SEP]"]
            )
 
        # Always create features DataFrame with age and abspos
        features_lf = tokenized_event_lf
        features_lf = create_abspos(features_lf)
        features_lf = create_age(lf=features_lf, birthdates=birthdates)
        if self.segment:
            features_lf = create_segment(features_lf)
        features_lf = features_lf.sort("date_col", maintain_order=True)
        features_lf = features_lf.drop(["date_col"])
 
        return features_lf
 
    @staticmethod
    def _load_if_exists(path: Path, lazy=True):
        if path.exists():
            print("Loading", path.stem)
            if path.suffix == ".parquet":
                if lazy:
                    return pl.scan_parquet(path)
                else:
                    return pl.read_parquet(path)
            elif path.suffix == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                raise ValueError("Only .parquet and .json files allowed")
        else:
            print("Creating", path.stem)
            return None
 
    def get_background_subset(
        self, sources: List[pl.LazyFrame], background: pl.DataFrame
    ) -> pl.DataFrame:
        """Load or subset background with pnrs of sources"""
        background_subsetted_path = (
            self.dir_path / "background_sample_inner_join.parquet"
        )
 
        # Load or subset background
        if (
            background_subsetted := self._load_if_exists(background_subsetted_path, lazy=False)
        ) is None:
            pnrs = get_pnrs(sources)
            background_subsetted = background.join(pnrs, on="person_id", how="inner")
            background_subsetted.write_parquet(background_subsetted_path)
        return background_subsetted
 
    def get_lazy_birthdates(self, background: pl.DataFrame) -> pl.LazyFrame:
        """Load or create the birthdates lazyframe"""
        birthdate_path = self.dir_path / "birthdates.parquet"
 
        # Load or create birthdates
        if (birthdates := self._load_if_exists(birthdate_path)) is None:
            birthdates = background.select("person_id", "birthday")
            birthdates.write_parquet(birthdate_path)
        return birthdates.lazy()
 
    def get_lazy_background(self, background_df: pl.DataFrame) -> pl.LazyFrame:
        """Load or create the background with all columns - ('person_id', 'birthday')"""
        background_path = self.dir_path / "background.parquet"
 
        if (background := self._load_if_exists(background_path)) is None:
            background = create_background(background_df)
            background.write_parquet(background_path)
        return background.lazy()
 
    def get_vocab(self, sources: List[pl.LazyFrame]) -> dict:
        """Load or create the vocabulary"""
        vocab_path = self.dir_path / "vocab.json"
 
        if (vocab := self._load_if_exists(vocab_path)) is None:
            vocab = create_vocab(sources)
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(vocab, f)
        return vocab
 
    def get_lazy_tokenized_event_lf(
        self, sources: List[pl.LazyFrame], vocab: dict
    ) -> pl.LazyFrame:
        """Load or create the tokenized event dataframe"""
 
        tokenized_path = self.dir_path / "tokenized.parquet"
 
        if (tokenized_event_lf := self._load_if_exists(tokenized_path)) is None:
            tokenized_event_lf = create_tokenized_events(
                sources, vocab, self.dir_path
            )  # tokenized_event_lf is saved within create_tokenized_events function
        return tokenized_event_lf