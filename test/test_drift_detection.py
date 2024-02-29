import numpy as np
import pytest
from drift_detection import DriftDetection

class TestDriftDetection:
    def test_check_labeling_consistency_with_conflict(self):
        """
        CHeck consistency
        """
        L = np.array([[1, 1, -1],
                      [1, -1, -1],
                      [-1, -1, 1]])
        
        dd = DriftDetection(conflict_threshold=-1)

        assert dd.check_labeling_consistency(L) == True

    def test_check_labeling_consistency_without_conflict(self):
        """
        Check consistency
        """
        L = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]])
        
        dd = DriftDetection(conflict_threshold=0.3)  # 设置冲突阈值为0.3
        
        assert dd.check_labeling_consistency(L) == False

    def test_check_for_virtual_drift_with_empty_model_window(self):
        """
        Check virtual drift for empty model window
        """
        dd = DriftDetection()
        model_window = []
        stream_window = [1, 2, 3]
        
        assert dd.check_for_virtual_drift(model_window, stream_window) == True

    def test_check_for_virtual_drift_with_mismatched_window_lengths(self):
        """
        Check virtual drift with mismatched window lengths
        """
        dd = DriftDetection()
        model_window = [1, 2, 3]
        stream_window = [1, 2] 
        
        with pytest.raises(ValueError):
            dd.check_for_virtual_drift(model_window, stream_window)
