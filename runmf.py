from pyspark import SparkConf, SparkContext
from bpr import optimize_mf

conf = (SparkConf().setMaster("local")
                   .setAppName("BPR")
                   .set("spark.executor.memory", "10g"))

sc = SparkContext(conf=conf)


if __name__ == '__main__':
    
    PREFIX = './'

    ratings = sc.textFile(
        "%straining_ratings.txt" % PREFIX
    ).map(
        lambda line: line.split(" ")
    ).map(
        lambda x: map(int, x[:2])
    )

    userMat, prodMat = optimizeMF(ratings, 20, 10)