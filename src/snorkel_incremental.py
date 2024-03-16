from typing import Any, Optional, List
import numpy as np
from numpy import ndarray
import torch
from snorkel.labeling.model import LabelModel
import logging

class IncLabelModel(LabelModel):
    """
    Incremental Label Model for semi-supervised learning.

    This class extends the base `LabelModel` class and provides additional functionality
    for incremental training and updating of the label model.

    Args:
        cardinality (int): The number of classes. Defaults to 2.
        **kwargs: Additional keyword arguments to be passed to the base `LabelModel` class.

    Attributes:
        n_total (int): The total number of training examples seen so far.

    Methods:
        _generate_O: Generates the matrix O used for training.
        fit: Fits the label model to the training data.
        incremental_fit: Performs incremental training on the label model.
    """

    def __init__(self, cardinality: int = 2, **kwargs: Any) -> None:
        super().__init__(cardinality, **kwargs)
        self.n_total = 0
    
    def _generate_O(self, L: ndarray, higher_order: bool = False, O_old: Optional[torch.Tensor] = None) -> None:
        """
        Generates the matrix O used for training.

        Args:
            L (ndarray): The label matrix.
            higher_order (bool): Whether to use higher-order dependencies. Defaults to False.
            O_old (Optional[torch.Tensor]): The previous matrix O. Defaults to None.
        """
        L_aug = self._get_augmented_label_matrix(L, higher_order=higher_order)
        self.d = L_aug.shape[1]
        O_new = torch.from_numpy(L_aug.T @ L_aug / L.shape[0]).float().to(self.config.device)
        
        if O_old is not None:
            # Calculate the weight of old and new O
            w_old = O_old.shape[0] / (O_old.shape[0] + L.shape[0])
            w_new = L.shape[0] / (O_old.shape[0] + L.shape[0])
            self.O = w_old * O_old + w_new * O_new
        else:
            self.O = O_new

    def fit(self, L_train: ndarray, Y_dev: ndarray | None = None, class_balance: List[float] | None = None, progress_bar: bool = True, **kwargs: Any) -> None:
        """
        Fits the label model to the training data.

        Args:
            L_train (ndarray): The training label matrix.
            Y_dev (ndarray | None): The development label matrix. Defaults to None.
            class_balance (List[float] | None): The class balance weights. Defaults to None.
            progress_bar (bool): Whether to display a progress bar. Defaults to True.
            **kwargs: Additional keyword arguments to be passed to the base `fit` method.
        """
        self.n_total += L_train.shape[0]
        super().fit(L_train, Y_dev, class_balance, progress_bar, **kwargs)

    def incremental_fit(
        self,
        L_train: ndarray, 
        Y_dev: Optional[ndarray] = None,
        class_balance: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Performs incremental training on the label model.

        Args:
            L_train (ndarray): The training label matrix.
            Y_dev (Optional[ndarray]): The development label matrix. Defaults to None.
            class_balance (Optional[List[float]]): The class balance weights. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the base `incremental_fit` method.
        """
        self.n_total += L_train.shape[0]

        L_shift = L_train + 1
        if L_shift.max() > self.cardinality:
            raise ValueError(
                f"L_train has cardinality {L_shift.max()}, but cardinality={self.cardinality} was passed in."
            )
        self._set_constants(L_shift)

        if class_balance is not None:
            self._set_class_balance(class_balance, Y_dev)
        
        self._generate_O(L_shift, O_old=self.O)
        
        # if self.train_config.mu_eps is None:
        #     self.train_config.mu_eps = min(0.01, 1 / 10 ** np.ceil(np.log10(self.n_total)))

        for key, value in kwargs.items():
            if hasattr(self.train_config, key):
                setattr(self.train_config, key, value)
        
        if hasattr(self, 'optimizer'):
            self._set_optimizer()  
        if hasattr(self, 'lr_scheduler'):
            self._set_lr_scheduler()

        self.train()
        self.to(self.config.device)
        
        for epoch in range(self.train_config.n_epochs):
            self.running_loss = 0.0
            self.running_examples = 0
            
            self.optimizer.zero_grad()
            loss = self._loss_mu(l2=self.train_config.l2)
            if torch.isnan(loss):
                logging.error("Loss is NaN. Consider reducing learning rate.")
                break
            loss.backward()
            self.optimizer.step()
            self._execute_logging(loss)
            self._update_lr_scheduler(epoch)

        self._clamp_params()
        self.eval()

        if self.config.verbose:
            logging.info("Finished Incremental Training")
