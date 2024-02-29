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
    """
    Processes a batch of messages, predicts their labels using a LabelModelActor, and sends the results to a Kafka topic.

    Args:
        batch_messages (List): A list of messages to process.
        label_model_actor (LabelModelActor): A Ray actor that encapsulates a Snorkel LabelModel for label prediction.
        kafka_producer (KafkaProducer): A Kafka producer for sending messages.
        output_topic (str): The Kafka topic to which the predictions are sent.
    """
    df = pd.DataFrame(batch_messages)
    predictions = ray.get(label_model_actor.predict.remote(df))
    df["pred_label"] = predictions
        
    for _, row in df.iterrows():
        msg = row.to_dict()
        kafka_producer.producer.send(output_topic, msg)

class SnorkelStreaming:
    """
    Implements a streaming processing system using Snorkel for data labeling and drift detection.

    Attributes:
        input_topic (str): The Kafka input topic from which messages are consumed.
        output_topic (str): The Kafka output topic to which labeled messages are produced.
        lfs (List[labeling_function]): A list of Snorkel labeling functions.
        bootstrap_servers (List[str]): A list of Kafka broker addresses.
        cardinality (int): The number of classes in the labeling task.
        batch_size (int): The number of messages to process in a batch.
        window_size (int): The size of the window for model and stream data, used in drift detection.
    """

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
        """
        Initializes the SnorkelStreaming object with Kafka topics, labeling functions, and configuration parameters.
        """
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
        """
        Starts the streaming process, including Kafka consumption and periodic drift checks.
        """
        self.log.info("Starting snorkel streaming...")
        self.should_stop = False
        Timer(60, self._drift_check).start(daemon=True)
        self._process_stream()
    
    def stop(self) -> None:
        """
        Stops the streaming process gracefully.
        """
        self.should_stop = True

    def _drift_check(self) -> None:
        """
        Periodically checks for drift in the data stream and triggers model retraining if necessary.
        """
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
        """
        Closes Kafka consumer and producer, terminating the streaming process.
        """
        self.consumer.close()
        self.producer.flush()
        self.producer.close()
        self.log.info("Stream processing terminated.")

    def _process_stream(self) -> None:
        """
        Continuously processes the incoming stream of messages from the input Kafka topic.
        """
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
        """
        Integrates new data into the model window and triggers retraining of the label model.

        Args:
            integration_ratio (float): The proportion of new data to integrate from the stream window.
        """
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
        """
        Triggers an alarm in response to detected drift or other significant events.
        """
        ...