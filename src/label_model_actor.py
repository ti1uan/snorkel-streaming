import ray
from typing import List, Deque
from snorkel.labeling.model import LabelModel
from snorkel.labeling import labeling_function, PandasLFApplier
import numpy as np
import pandas as pd

@ray.remote
class LabelModelActor:
    """
    A Ray actor class for applying labeling functions, training, and predicting with a Snorkel LabelModel.

    Attributes:
        label_model (LabelModel): The Snorkel LabelModel used for training and prediction.
        applier (PandasLFApplier): Applier to apply labeling functions over the data.
    """
    
    def __init__(self, lfs: List[labeling_function], cardinality: int) -> None:
        """
        Initializes the LabelModelActor with labeling functions and cardinality for the LabelModel.

        Args:
            lfs (List[labeling_function]): A list of Snorkel labeling functions to be applied.
            cardinality (int): The number of classes or categories in the labeling task.
        """
        self.label_model = LabelModel(cardinality=cardinality)
        self.applier = PandasLFApplier(lfs=lfs)
    
    def apply_lfs(self, data: Deque) -> np.ndarray:
        """
        Applies the labeling functions to the provided data and returns the labeling matrix.

        Args:
            data (Deque): A deque of data points (typically pandas DataFrame rows) to which labeling functions will be applied.

        Returns:
            np.ndarray: A labeling matrix where rows correspond to data points and columns correspond to labeling function outputs.
        """
        return self.applier.apply(df=pd.DataFrame(list(data)))

    def train_model(self, data: Deque) -> None:
        """
        Trains the LabelModel using the provided data.

        Applies labeling functions to the data, then fits the LabelModel to the resulting labeling matrix.

        Args:
            data (Deque): A deque of data points used for training the model.
        """
        L = self.apply_lfs(data)
        self.label_model.fit(L, n_epochs=500, log_freq=100, seed=123)

    def predict(self, data) -> np.ndarray:
        """
        Predicts the labels for the given data using the trained LabelModel.

        Before prediction, it checks for virtual drift between the model and stream data windows. If drift is detected, it integrates new data and updates the model.

        Args:
            data (Deque): A deque of data points for which to predict labels.

        Returns:
            np.ndarray: An array of predicted labels.
        """
        L = self.apply_lfs(data)
        predictions = self.label_model.predict(L)
        return predictions