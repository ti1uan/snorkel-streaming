from pyflink.datastream import StreamExecutionEnvironment, DataStream
from pyflink.datastream.functions import MapFunction
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream import StreamExecutionEnvironment, DataStream
from pyflink.datastream.functions import MapFunction
from snorkel.labeling import labeling_function
import os
import time

#labeling function 1
@labeling_function()
def is_positive(x):
    return 1 if x%2==0 else 0

#labeling function 2
@labeling_function()
def is_negative(x):
    return 1 if x%3==0  else 0

lfs = [is_positive, is_negative]

class SnorkelLabelingOperator(MapFunction):
    def __init__(self, lfs, cardinality):
        self.lfs = lfs
        self.cardinality = cardinality
        self.label_model = None 

    def map(self, value):
        #For each element, label with the Snorkel labeling function
        #Here, we assume that the label model has already been trained and can be used directly
        #In practice, it may be necessary to dynamically train or update the model based on new data
        predictions = [lf(value) for lf in self.lfs]
        labels = {str(i): float(pred) for i, pred in enumerate(predictions)}
        ss =  f"{value} (label: {max(labels, key=labels.get)})"
        # print(ss)
        # print(labels)
        return ss

    def open(self, config):

        self.label_model = "some_label_model"

    def close(self, config):
        # clean rescource
        pass



def main():
    # set flink cluster
    os.environ['JOB_MANAGER_RPC_ADDRESS'] = 'flink-jobmanager'
    
    # get environment
    env = StreamExecutionEnvironment.get_execution_environment()

    # set palllism
    env.set_parallelism(1)  # based on the number of TaskManager

    # create strem
    data_stream = env.from_collection(range(800000)).map(lambda x: x)
    print(type(data_stream))
    # apply snrokel operator
    snorkel_operator = SnorkelLabelingOperator(lfs,2)
    labeled_stream = data_stream.map(snorkel_operator)

   
    start_time = time.time()
    env.execute("Test Flink Job")  # start flink

    end_time = time.time()

    
    processing_time = end_time - start_time
    print(f"Processing time: {processing_time} seconds")

if __name__ == '__main__':
    main()

