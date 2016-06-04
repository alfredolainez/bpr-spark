import java.io.File
import scala.util.Random

import org.apache.spark.{RangePartitioner, SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.optimization.{Gradient, GradientDescent}
import org.apache.spark.mllib.random.RandomRDDs._
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD

import breeze.optimize.linear._
import breeze.numerics._
import breeze.util.partition
import breeze.linalg.DenseMatrix

object MainBPR {

  def main(args: Array[String]) {

    object BPR extends Serializable {

      private def gradientSinglePoint(userId: Int, prodPos: Int, prodNeg: Int,
                                      userMat: DenseMatrix[Double], prodMat: DenseMatrix[Double],
                                      lambdaReg: Double = 0.01, alpha: Double = 0.1): Unit = {

        val x_uij = userMat(userId, ::).dot(prodMat(prodPos, ::)) - userMat(userId, ::).dot(prodMat(prodNeg, ::))

        val scale = math.exp(-x_uij) / (1 + math.exp(-x_uij))
        userMat(userId, ::) :+= ((prodMat(prodPos, ::) - prodMat(prodNeg, ::)) :* scale) :* alpha
        prodMat(prodPos, ::) :+= (userMat(userId, ::) :* scale) :* alpha
        prodMat(prodNeg, ::) :+= (-userMat(userId, ::) :* scale) :* alpha

        // Add regularization
        //userMat :+= userMat :* (lambdaReg * alpha)
        //prodMat :+= prodMat :* (lambdaReg * alpha)

      }

      private def sampleAndOptimizePartition(ratings: Iterator[(Int, Int)], userMat: DenseMatrix[Double],
                                             prodMat: DenseMatrix[Double], numProds: Int, lambdaReg: Double = 0.1,
                                             alpha: Double = 0.01): Iterator[Tuple2[DenseMatrix[Double], DenseMatrix[Double]]] = {

        val NUM_OF_NEGATIVE_PER_IMPLICIT = 30

        val positiveRatingsRepeated = ratings.flatMap(x => Vector.fill(NUM_OF_NEGATIVE_PER_IMPLICIT)(x)).toVector
        val negativeRatings = positiveRatingsRepeated.map(x => (x._1, x._2, Random.nextInt(numProds) + 1))

        //val allRatings = new Vector[(Int, Int, Int)](negativeRatings) // TODO: REMOVE THE NEGATIVE STUFF THAT IS NOT ACTUALLY NEGATIVE SAMPLING?
        val allRatings = negativeRatings
        val sampledRatings = Random.shuffle(allRatings.toList).toVector.slice(0, 20000) // TODO: ADD PROPER CONSTANTS

        for (sampledPoint <- sampledRatings) {
          gradientSinglePoint(sampledPoint._1, sampledPoint._2, sampledPoint._3, userMat, prodMat)
        }

        return List((userMat, prodMat)).iterator
      }

      def optimizeMF(ratings: RDD[(Int, Int)], rank: Int = 10,
                     numIterations: Int = 10): (DenseMatrix[Double], DenseMatrix[Double]) = {

        // Partition by user
        val userPartitioner = new RangePartitioner(4, ratings)
        val ratingsPartitioned = ratings.partitionBy(userPartitioner).persist()

        val numUsers = ratingsPartitioned.map(x => x._1).max()
        val numSongs = ratingsPartitioned.map(x => x._2).max()

        var userMat: DenseMatrix[Double] = DenseMatrix.rand[Double](numUsers + 1, rank)
        var prodMat: DenseMatrix[Double] = DenseMatrix.rand[Double](numSongs + 1, rank)

        for (i <- 1 until numIterations) {
          val result = ratingsPartitioned.mapPartitions {
            ratings => sampleAndOptimizePartition(ratings, userMat, prodMat, numSongs)
          }

          // Average through parameters
          val num = result.count.toDouble
          val result2 = result.reduce((a, b) => (a._1 + b._1, a._2 + b._2))

          userMat = result2._1 :/ num
          prodMat = result2._2 :/ num
        }

        return (userMat, prodMat)
      }

    }

    val conf = new SparkConf().setAppName("BPR").setMaster("local")
    val sc = new SparkContext(conf)

    val ratingsBPR = sc.textFile("/home/alfredo/Desktop/bpr/training_ratings.txt").map(line => line.split(" ")).map(x => (x(0).toInt, x(1).toInt))
    val (userMat, prodMat) = BPR.optimizeMF(ratingsBPR, 10, 20)

    breeze.linalg.csvwrite(new File("/home/alfredo/Desktop/bpr/userMatrix.txt"), userMat, separator = ' ')
    breeze.linalg.csvwrite(new File("/home/alfredo/Desktop/bpr/prodMatrix.txt"), prodMat, separator = ' ')

    // WITH ALS
    val ratings = sc.textFile("/home/alfredo/Desktop/bpr/training_ratings.txt").map(line => line.split(" ")).map(x => (x(0).toInt, x(1).toInt, 1))

    val rank = 10
    val numIterations = 10
    val ALSRatings = ratings.map{ case (user, item, num) => Rating(user.toInt, item.toInt, num.toDouble)}
    val model = ALS.trainImplicit(ALSRatings, rank, numIterations, 0.01, 0.01)

    model.productFeatures.saveAsTextFile("/home/alfredo/Desktop/bpr/als_product_matrix.txt")
    model.userFeatures.saveAsTextFile("/home/alfredo/Desktop/bpr/als_user_matrix.txt")

    // Evaluate the model on rating data
    val usersProducts = ALSRatings.map { case Rating(user, product, rate) =>
      (user, product)
    }
    val predictions =
      model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }
    val ratesAndPreds = ALSRatings.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions)
    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()
    println("Mean Squared Error = " + MSE)



  }
}
