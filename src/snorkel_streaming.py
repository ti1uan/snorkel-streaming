from kafka import KafkaConsumer, KafkaProducer
import json
import pandas as pd
from scipy.spatial.distance import jensenshannon
from collections import deque
from snorkel.labeling import labeling_function
from typing import List
from drift_detection import DriftDetection
import logging
import ray
from threading import Timer
from label_model_actor import LabelModelActor

@ray.remote
def process_batch(
    batch_messages: List,
    label_model_actor: LabelModelActor,
    kafka_producer: KafkaProducer,
    output_topic: str
) -> None:
    df = pd.DataFrame(batch_messages)
    predictions = ray.get(label_model_actor.predict.remote(df))
    df["pred_label"] = predictions
        
    for _, row in df.iterrows():
        msg = row.to_dict()
        kafka_producer.producer.send(output_topic, msg)

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
        self.label_model_actor = LabelModelActor(lfs, cardinality)
        self.drift_detection = DriftDetection()
        self.batch_size = batch_size
        self.output_topic = output_topic
        self.model_window = deque(maxlen=window_size)
        self.stream_window = deque(maxlen=window_size)

        self.should_train = False
        self.should_stop = None
        self.log = logging.getLogger(__name__)
    
    def start(self) -> None:
        self.log.info("Starting snorkel streaming...")
        self.should_stop = False
        Timer(60, self._drift_check).start(daemon=True)
        self._process_stream()
    
    def stop(self) -> None:
        self.should_stop = True

    def _drift_check(self) -> None:
        if self.should_stop:
            return

        if self.model_window == self.model_window.maxlen:
            if self.drift_detection.check_for_virtual_drift(self.model_window, self.stream_window):
                self.should_train = True
            
            L = ray.get(self.label_model_actor.apply_lfs.remote(self.stream_window))
            if self.drift_detection.check_labeling_consistency(L=L):
                self._trigger_alarm()

        Timer(60, self._drift_check).start(daemon=True)


    def _terminate(self) -> None:
        self.consumer.close()
        self.producer.flush()
        self.producer.close()
        self.log.info("Stream processing terminated.")

    def _process_stream(self) -> None:
        batch_messages = []
        for message in self.consumer:
            if not self.should_train:
                # Initialize model
                self.model_window.append(message.value)
                if len(self.model_window) == self.model_window.maxlen:
                    self.log.info("Initial training for label model")
                    self.label_model_actor.train_model.remote(self.model_window)
                    self.should_train = True
                    self.log.info("Initial label model trained")
                else:
                    continue

            batch_messages.append(message.value)
            self.stream_window.append(message.value)

            if len(batch_messages) >= self.batch_size:
                process_batch.remote(batch_messages, self.label_model_actor, self.producer, self.output_topic)
                batch_messages = []
            
            if self.should_stop:
                break
        if len(batch_messages) > 0:
            process_batch.remote(batch_messages, self.label_model_actor, self.producer, self.output_topic)
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
        
        self.label_model_actor.train_model.remote(self.model_window)
    
    def _trigger_alarm(self) -> None:
        ...