import pytest
import ray
import numpy as np
import pandas as pd
from snorkel.labeling import labeling_function
from snorkel.labeling.apply.pandas import PandasLFApplier

from label_model_actor import LabelModelActor  # Adjust this import based on your file structure

ray.init(ignore_reinit_error=True)

# Define dummy labeling functions for testing
@labeling_function()
def lf1(x):
    return 1 if x['feature'] > 0 else 0

@labeling_function()
def lf2(x):
    return 1 if x['feature'] < 0 else 0

@labeling_function()
def lf3(x):
    return 1 if x['feature'] == 0 else 0

@pytest.fixture
def sample_data():
    # Create a simple pandas DataFrame to simulate data
    return pd.DataFrame({'feature': [-2, -1, 0, 1, 2]})

@pytest.fixture
def label_model_actor():
    # Initialize and return a LabelModelActor instance
    actor = LabelModelActor.remote([lf1, lf2, lf3], cardinality=2)
    return actor

def test_apply_lfs(sample_data, label_model_actor):
    # Test applying labeling functions to data
    L_matrix_ref = label_model_actor.apply_lfs.remote(sample_data.to_dict('records'))
    L_matrix = ray.get(L_matrix_ref)
    assert isinstance(L_matrix, np.ndarray)
    assert L_matrix.shape == (5, 3)  # Expecting a matrix with 5 samples and 2 labeling functions

def test_train_model(sample_data, label_model_actor):
    # Test training the LabelModel
    train_model_ref = label_model_actor.train_model.remote(sample_data.to_dict('records'))
    # Wait for the training to complete
    ray.get(train_model_ref)
    # This test assumes successful execution of training. For more detailed testing, consider inspecting the state of the model.

def test_predict(sample_data, label_model_actor):
    # Make sure train first
    train_model_ref = label_model_actor.train_model.remote(sample_data.to_dict('records'))
    # Wait for the training to complete
    ray.get(train_model_ref)

    # Test predicting labels for data
    predictions_ref = label_model_actor.predict.remote(sample_data.to_dict('records'))
    predictions = ray.get(predictions_ref)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 5  # Expecting predictions for 5 samples
