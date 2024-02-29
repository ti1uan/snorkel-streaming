import ray
from typing import List, Deque
from snorkel.labeling.model import LabelModel
from snorkel.labeling import labeling_function, PandasLFApplier
import numpy as np
import pandas as pd

@ray.remote
class LabelModelActor:
    def __init__(self, lfs: List[labeling_function], cardinality: int) -> None:
        self.label_model = LabelModel(cardinality=cardinality)
        self.applier = PandasLFApplier(lfs=lfs)
    
    def apply_lfs(self, data: Deque) -> np.ndarray:
        return self.applier.apply(df=pd.DataFrame(list(data)))

    def train_model(self, data: Deque) -> None:
        L = self.apply_lfs(data)
        self.label_model.fit(L, n_epochs=500, log_freq=100, seed=123)

    def predict(self, data) -> np.ndarray:
        L = self.apply_lfs(data)

        # Check data distribution for model window and stream window for virtual drift
        if self.drift_detection.check_for_virtual_drift(self.model_window, self.stream_window):
            self._integrate_new_data_and_update_model()

        predictions = self.label_model.predict(L)
        return predictions