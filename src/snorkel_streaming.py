from snorkel.labeling import LFAnalysis, PandasLFApplier
from snorkel.labeling.model import LabelModel
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
import json
import pandas as pd
from scipy.spatial.distance import jensenshannon
from collections import deque
from snorkel.labeling import labeling_function
from typing import List, Deque
from drift_detection import DriftDetection
import logging

class SnorkelStreaming:
    def __init__(
        self,
        input_topic: str,
        output_topic: str,
        lfs: List[labeling_function],
        bootstrap_servers: List[str]=['localhost:9092'],
        cardinality: int=2,
        batch_size: int=100,
        window_size: int=1000
    ) -> None:
        if len(lfs) < 3:
            raise ValueError("At least three labeling functions are required")

        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=bootstrap_servers, 
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        self.label_model = LabelModel(cardinality=cardinality)
        self.applier = PandasLFApplier(lfs=lfs)
        self.drift_detection = DriftDetection()
        self.batch_size = batch_size
        self.output_topic = output_topic
        self.model_window = deque(maxlen=window_size)
        self.stream_window = deque(maxlen=window_size)

        self.initialized = False
        self.should_stop = None
        self.log = logging.getLogger(self.__class__.__name__)
    
    def start(self) -> None:
        self.log.info("Starting snorkel streaming...")
        self.should_stop = False
        self._process_stream()
    
    def stop(self) -> None:
        self.should_stop = True

    def _terminate(self) -> None:
        self.consumer.close()
        self.producer.flush()
        self.producer.close()
        self.log.info("Stream processing terminated.")

    def _process_stream(self) -> None:
        batch_messages = []
        for message in self.consumer:
            if not self.initialized:
                # Initialize model
                self.model_window.append(message.value)
                if len(self.model_window) == self.batch_size:
                    self.log.info("Initial training for label model")
                    self._update_model(self.model_window)
                    self.initialized = True
                    self.log.info("Initial label model trained")
                else:
                    continue

            batch_messages.append(message.value)
            self.stream_window.append(message.value)

            if len(batch_messages) >= self.batch_size:
                self._process_batch(batch_messages)
                batch_messages = []
            
            if self.should_stop:
                break
        if len(batch_messages) > 0:
            self._process_batch(batch_messages)
        self._terminate()
    
    def _integrate_new_data_and_update_model(self, integration_ratio: float=0.3) -> None:
        # Calculate the number of integrating data.
        num_new_data = int(len(self.stream_window) * integration_ratio)
        
        # Fetch the latest data from streaming window
        new_data = list(self.stream_window)[-num_new_data:]
        
        # Update model window
        for data in new_data:
            if len(self.model_window) == self.model_window.maxlen:
                self.model_window.popleft()
            self.model_window.append(data)
        
        self._update_model(self.model_window)

    def _update_model(self, data: Deque) -> None:
        # Apply labeling functions
        df = pd.DataFrame(list(data))
        L = self.applier.apply(df=df)
        # Retrain the label model
        self.label_model.fit(L, n_epochs=500, log_freq=100, seed=123)

    def _process_batch(self, batch_messages: List) -> None:
        df = pd.DataFrame(batch_messages)
        L_batch = self.applier.apply(df=df)
        
        # Check data distribution for model window and stream window for virtual drift
        if self.drift_detection.check_for_virtual_drift(self.model_window, self.stream_window):
            self._integrate_new_data_and_update_model()

        # Check LF consistency for real drift
        if self.drift_detection.check_labeling_consistency(L_batch):
            self.trigger_alarm()

        df = pd.DataFrame(batch_messages)
        L = self.applier.apply(df=df)
        df['pred_label'] = self.label_model.predict(L=L)
        
        for _, row in df.iterrows():
            msg = row.to_dict()
            try:
                self.producer.send(self.output_topic, msg)
            except Exception as e:
                self.log.error(f"Error occurred when try produce label: {e}")
                self._terminate()
    
    def _trigger_alarm(self) -> None:
        ...