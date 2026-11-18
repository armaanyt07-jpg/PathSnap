"""Fitted emission scorer and transition settings for a map-matching HMM."""
from dataclasses import dataclass

from .decoder import Config
from pathsnap.features import HMM_FEATURE_COLUMNS


@dataclass
class TrainedHMM:
    emission_model: object
    decoder_config: Config
    feature_columns: tuple[str, ...] = tuple(HMM_FEATURE_COLUMNS)

    def predict_proba(self, X):
        return self.emission_model.predict_proba(X)
