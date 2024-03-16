from snorkel.labeling import labeling_function, PandasLFApplier, LFApplier
from snorkel.labeling.model import LabelModel
from collections import deque
import pandas as pd
import numpy as np
from typing import List, Callable, Union, Sequence, Any
from scipy.stats import mode

class MetaLabelModel:
    """
    A meta-label model that combines multiple LabelModels for prediction.

    Args:
        cardinality (int): The number of classes for the label.
        lfs (List[Callable]): List of labeling functions to be applied.
        max_models (int, optional): Maximum number of LabelModels to keep. Defaults to 3.
        data_mode (str, optional): Data mode, either 'seq' for sequence data or 'df' for DataFrame data. Defaults to 'seq'.
        n_epochs (int, optional): Number of training epochs for each LabelModel. Defaults to 500.
    """

    def __init__(
        self,
        cardinality: int,
        lfs: List[Callable],
        max_models: int=3,
        data_mode: str='seq',
        n_epochs: int=500
    ) -> None:
        assert data_mode in ['seq', 'df']
        self.applier = LFApplier(lfs=lfs) if data_mode == 'seq' else PandasLFApplier(lfs=lfs)
        self.cardinality = cardinality
        assert max_models >= 3, "max_model must be at least 3"
        self.models = deque(maxlen=max_models)
        self.data_mode = data_mode
        self.n_epochs = n_epochs
        self.meta_model = None
        self.meta_applier = None
        self.mode = 'vote'

    def update(self, data: Union[Sequence[Any], pd.DataFrame]) -> None:
        """
        Update the meta-label model with new data.

        Args:
            data (Union[Sequence[Any], pd.DataFrame]): The input data for updating the model.
        """
        assert self.data_mode == 'seq' or isinstance(data, pd.DataFrame), \
            "Data must be a DataFrame when using dataframe mode"
        
        # Add new model to models queue
        new_model = LabelModel(cardinality=self.cardinality)
        L_train = self.applier.apply(data)
        new_model.fit(L_train=L_train, n_epochs=self.n_epochs)

        self.models.append(new_model)
        if len(self.models) >= 3:
            self.mode = 'meta'
            self._train_meta_model(data)

    def predict(self, data: Union[Sequence[Any], pd.DataFrame]) -> np.ndarray:
        """
        Predict labels for the input data.

        Args:
            data (Union[Sequence[Any], pd.DataFrame]): The input data for prediction.

        Returns:
            np.ndarray: The predicted labels.
        """
        assert len(self.models) > 0, "No models available for prediction"
        assert self.data_mode == 'seq' or isinstance(data, pd.DataFrame), \
            "Data must be a DataFrame when using dataframe mode"

        if self.mode == 'vote':
            L = self.applier.apply(data)
            predictions = []
            for m in self.models:
                predictions.append(m.predict(L))
            predictions = np.vstack(predictions)
            mode_result, _ = mode(predictions, axis=0)
            final_predictions = mode_result.squeeze()
            return final_predictions
        
        elif self.mode == 'meta':
            assert len(self.models) > 3, "There must be at least three models when using meta"
            assert self.meta_model and self.meta_applier, "Meta model not trained yet."

            L = self.meta_applier.apply(data)
            return self.meta_model.predict(L)
        
        else:
            raise ValueError('Invalid mode')
        
    def _train_meta_model(self, data: Union[Sequence[Any], pd.DataFrame]) -> None:
        """
        Train the meta-label model using the existing LabelModels.

        Args:
            data (Union[Sequence[Any], pd.DataFrame]): The input data for training the meta-label model.
        """

        def _make_labeling_function(model):
            @labeling_function()
            def lf(x):
                return model.predict(x)
            return lf
        
        lfs = [_make_labeling_function(model) for model in self.models]
        self.meta_applier = LFApplier(lfs=lfs) if self.data_mode == 'seq' else PandasLFApplier(lfs=lfs)
        self.meta_model = LabelModel(cardinality=self.cardinality)

        L_train = self.meta_applier.apply(data)
        self.meta_model.fit(L_train=L_train, n_epochs=self.n_epochs)