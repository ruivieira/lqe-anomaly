import os
import collections
import argparse


os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.1.0 pyspark-shell'

parser = argparse.ArgumentParser(description='LQE Anomaly detection from sensor readings')
parser.add_argument('--servers', help='The bootstrap servers', default='localhost:9092')
parser.add_argument('--topic', help='Topic to consume', default='anomaly')
parser.add_argument('--nsensors', help='Number of sensors', default=10)
parser.add_argument('--history', help='History size', default=50)

args = parser.parse_args()

SERVER = args.servers
TOPIC = args.topic
NSENSORS = args.nsensors
HISTORY = args.history

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import math

sc = SparkContext(appName="anomalysingle")
sc.setLogLevel("ERROR")

ssc = StreamingContext(sc, 5)
ssc.checkpoint('/tmp')

# consume sensor data from Kafka
directKafkaStream = KafkaUtils.createDirectStream(ssc, [TOPIC], {"metadata.broker.list": SERVER})


# group observations by sensor id
data = directKafkaStream.map(lambda x: (int(x[0]), float(x[1]))) \
	.groupByKey() \
	.map(lambda x: (x[0], list(x[1])))

# Stateful transformation to recursively estimate state
# so that discrepancy can be calculated
def kalman(values, state):
	observations = values[0]
	n = len(observations)
	
	# Prior Kalman moments and discrepancy
	ms = [0]
	Cs = [10]
	discrepancies = [0]

	# if we already have calculated states,
	# discard all except the last one
	if state is not None: 
		ms = [state[0][-1]]
		Cs = [state[1][-1]]
		discrepancies = [state[2][-1]]

	for y in observations:

		R = Cs[-1] + 1.0
		e = y - ms[-1]
		Q = R + 1.0
		K = R / Q

		m = ms[-1] + K * e
		C = R - Q * (K**2)

		discrepancy = abs(y - m)

		ms.append(m)
		Cs.append(C)
		discrepancies.append(discrepancy)

	# discard the initial state
	state = (ms[1:], Cs[1:], discrepancies[1:])

	return state

filtered = data.updateStateByKey(kalman)

# Temporary value storage (for plot usage)
obs = {}
discrepancies = {}
for i in xrange(NSENSORS):
    obs[i] = collections.deque(HISTORY*[0], HISTORY)
    discrepancies[i] = collections.deque(HISTORY*[0], HISTORY)

# collect observations for plotting
def store_obs(x):

	global obs
	collected = x.collect()

	for sensor in collected:
		# store observations
		id = sensor[0]
		ys = sensor[1]
		obs[id].extend(ys)
		
# collect discrepancies for plotting
def store_discrepancies(x):
	
	global discrepancies
	collected = x.collect()

	for sensor in collected:
		id = sensor[0]
		filtered = sensor[1]
		discrepancies[id].extend(filtered[2])

data.foreachRDD(store_obs)
filtered.foreachRDD(store_discrepancies)

ssc.start()

from flask import Flask, render_template
import json

app = Flask(__name__)

@app.route("/data")
def data():
	data = {}
	ts = xrange(HISTORY)
	for sensor in xrange(NSENSORS):
		data[sensor] = {}
		data[sensor]['data'] = {}
		data[sensor]['data'] =  [[{"x" : t, "y" : obs[sensor][t]} for t in ts], [{"x" : t, "y" : discrepancies[sensor][t]} for t in ts]]
		data[sensor]['outliers'] = []
		for t in xrange(HISTORY):
			if discrepancies[sensor][t] > 3.0: # outlier threshold set at 3 sigma
				data[sensor]['outliers'].append({'x': t, 'label' : ""})
	return json.dumps(data)

@app.route("/")
def main():
	 return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=False)


ssc.awaitTermination()

