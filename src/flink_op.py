from pyflink.datastream import StreamExecutionEnvironment, DataStream
from pyflink.datastream.functions import MapFunction, FilterFunction
import pandas as pd

# Define a MapFunction to process card transaction records
class ProcessCardTransaction(MapFunction):
    def map(self, value):
        # The map function simply passes through the value
        return value

# Define a FilterFunction to filter out fraudulent transactions
class FilterFraudTransactions(FilterFunction):
    def filter(self, value):
        # The filter function checks if the 'fraud' field is equal to 1.0
        return value['fraud'] == 1.0

# Get the execution environment from PyFlink
env = StreamExecutionEnvironment.get_execution_environment()

# Set the parallelism level for the job
env.set_parallelism(1)  # Set to 1 for this example

# Read the CSV file into a pandas DataFrame
filename = '/content/drive/My Drive/card_transdata.csv'
df = pd.read_csv(filename)
df = df.head(100)  # Select only the first 100 rows for this example

# Convert the DataFrame to a stream of data for processing
data_stream = env.from_collection(df.to_dict('records'))

# Process the data using the map function defined earlier
processed_stream = data_stream.map(ProcessCardTransaction())

# Filter the transactions to get only the fraudulent ones
fraud_stream = processed_stream.filter(FilterFraudTransactions())

# Execute the Flink job and collect the results
fraud_results = fraud_stream.execute_and_collect()

# Convert the filtered results back into a pandas DataFrame
fraud_df = pd.DataFrame(fraud_results)

# Print the head of the DataFrame to see the first few rows
print(fraud_df.head())

# Note: In an actual Flink job, you would start the job with the following command:
# env.execute("Card Transaction Processing with PyFlink")
