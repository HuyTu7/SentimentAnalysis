from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib.pyplot as plt


def main():
    conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec
    ssc.checkpoint("checkpoint")

    pwords = load_wordlist("positive.txt")
    nwords = load_wordlist("negative.txt")
   
    counts = stream(ssc, pwords, nwords, 100)
    make_plot(counts)


def make_plot(counts):
    """
    Plot the counts for the positive and negative words for each timestep.
    Use plt.show() so that the plot will popup.
    """
    # YOUR CODE HERE
    positives = np.array([entry[0][1] for entry in counts if entry > 0])
    negatives = np.array([entry[1][1] for entry in counts if entry > 0])
    p = plt.subplot(1, 1, 1)
    axis = range(0, len(positives))
    p.plot(axis, positives, 'bs-', label = "positive")
    p.plot(axis, negatives, 'rs-', label = "negative")
    p.set_ylim([0, max(max(positives), max(negatives)) + 100])

    plt.xlabel("Time step")
    plt.ylabel("Word count")
    plt.legend(fontsize = 'small', loc=0)
    plt.savefig("plot.png")
    plt.show()


def load_wordlist(filename):
    """ 
    This function should return a list or set of words from the given filename.
    """
    # YOUR CODE HERE
    with open(filename, 'r') as f:
        words = f.read().split('\n')
    return set(words)

def count_words(tweet, pos, neg):
    n_count, p_count = 0, 0
    words = set([word.lower() for word in tweet.split(' ')]) 
    n_count, p_count = len(neg.intersection(words)), len(pos.intersection(words))
    return [("positive", p_count), ("negative", n_count)]

def stream(ssc, pwords, nwords, duration):
    
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1].encode("ascii","ignore"))
    #tweets.pprint()
    # Each element of tweets will be the text of a tweet.
    # You need to find the count of all the positive and negative words in these tweets.
    # Keep track of a running total counts and print this at every time step (use the pprint function).
    # YOUR CODE HERE
    
    words = tweets.flatMap(lambda x: count_words(x, pwords, nwords))
    words = words.reduceByKey(lambda x, y: x + y)
    words.pprint()
    
    # Let the counts variable hold the word counts for all time steps
    # You will need to use the foreachRDD function.
    # For our implementation, counts looked like:
    #   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
    counts = []
    words.foreachRDD(lambda t, rdd: counts.append(rdd.collect())) 
    # YOURDSTREAMOBJECT.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    
    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)

    return counts


if __name__=="__main__":
    main()
