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
import breeze.linalg.{DenseMatrix, DenseVector}

object MainBPR {

  // TODO: ADD REGULARIZATION

  def main(args: Array[String]) {

    object BPR extends Serializable {

      private def gradientSinglePoint(userId: Int, prodPos: Int, prodNeg: Int,
                                      userMat: DenseMatrix[Double], prodMat: DenseMatrix[Double],
                                      lambdaReg: Double = 0.01, alpha: Double = 0.1): Unit = {

        val x_uij = userMat(userId, ::).dot(prodMat(prodPos, ::)) - userMat(userId, ::).dot(prodMat(prodNeg, ::))

        val scale = math.exp(-x_uij) / (1 + math.exp(-x_uij))
        prodMat(prodPos, ::) :+= ((userMat(userId, ::) :* scale) + (prodMat(prodPos, ::) :* lambdaReg)) :* alpha
        prodMat(prodNeg, ::) :+= ((-userMat(userId, ::) :* scale) + (prodMat(prodNeg, ::) :* lambdaReg)) :* alpha
        userMat(userId, ::) :+= (((prodMat(prodPos, ::) - prodMat(prodNeg, ::)) :* scale) +
          (userMat(userId, ::) :* lambdaReg)) :* alpha
      }

      private def sampleAndOptimizePartition(ratings: Iterator[(Int, Int)], userMat: DenseMatrix[Double],
                                             prodMat: DenseMatrix[Double], numProds: Int, numSamples: Int = 20000,
                                             lambdaReg: Double = 0.1, alpha: Double = 0.01): Iterator[Tuple2[DenseMatrix[Double], DenseMatrix[Double]]] = {

        val NUM_OF_NEGATIVE_PER_IMPLICIT = 5

        val positiveRatingsRepeated = ratings.flatMap(x => Vector.fill(NUM_OF_NEGATIVE_PER_IMPLICIT)(x)).toVector
        val negativeRatings = positiveRatingsRepeated.map(x => (x._1, x._2, Random.nextInt(numProds) + 1))

        val sampledRatings = Random.shuffle(negativeRatings.toList).toVector.slice(0, numSamples)

        for (sampledPoint <- sampledRatings) {
          gradientSinglePoint(sampledPoint._1, sampledPoint._2, sampledPoint._3, userMat, prodMat)
        }

        return List((userMat, prodMat)).iterator
      }

      def optimizeMF(ratings: RDD[(Int, Int)], rank: Int = 10,
                     numIterations: Int = 10, numPartitions: Int = 4): (DenseMatrix[Double], DenseMatrix[Double]) = {

        // Partition by user
        val userPartitioner = new RangePartitioner(4, ratings)
        val ratingsPartitioned = ratings.partitionBy(userPartitioner).persist()

        val numUsers = ratingsPartitioned.map(x => x._1).max()
        val numProds = ratingsPartitioned.map(x => x._2).max()

        var userMat: DenseMatrix[Double] = DenseMatrix.rand[Double](numUsers + 1, rank)
        var prodMat: DenseMatrix[Double] = DenseMatrix.rand[Double](numProds + 1, rank)

        for (i <- 1 until numIterations) {
          val result = ratingsPartitioned.mapPartitions {
            ratings => sampleAndOptimizePartition(ratings, userMat, prodMat, numProds)
          }

          // Average through parameters
          val numReducers = result.count.toDouble
          val averagedMatrices = result.reduce((a, b) => (a._1 + b._1, a._2 + b._2))

          userMat = averagedMatrices._1 :/ numReducers
          prodMat = averagedMatrices._2 :/ numReducers
        }

        return (userMat, prodMat)
      }

    }


    object DistributedUserBPR extends Serializable {

      private def gradientSinglePoint(prodPos: Int, prodNeg: Int,
                                      userVector: DenseVector[Double], prodMat: DenseMatrix[Double],
                                      lambdaReg: Double = 0.01, alpha: Double = 0.1): Unit = {

        val x_uij = userVector.dot(prodMat(prodPos, ::).t) - userVector.dot(prodMat(prodNeg, ::).t)

        val scale = math.exp(-x_uij) / (1 + math.exp(-x_uij))
        userVector :+= ((prodMat(prodPos, ::) - prodMat(prodNeg, ::)) :* scale).t :* alpha
        prodMat(prodPos, ::) :+= (userVector :* scale).t :* alpha
        prodMat(prodNeg, ::) :+= (userVector :* scale).t :* alpha

        // Add regularization
        //userMat :+= userMat :* (lambdaReg * alpha)
        //prodMat :+= prodMat :* (lambdaReg * alpha)

      }

      private def sampleAndOptimizePartition(userRatingsFeatures: Iterator[(Int, (Iterable[Int], DenseVector[Double]))],
                                             prodMat: DenseMatrix[Double], numProds: Int, lambdaReg: Double = 0.1,
                                             alpha: Double = 0.01): Iterator[(DenseMatrix[Double], Array[(Int, DenseVector[Double])])] = {

        val NUM_OF_NEGATIVE_PER_IMPLICIT = 30

//        def getProds(userId: Int, productList: Iterable[Int]) : Iterable[(Int,Int)] = {
//          val userRatings = productList.map(prod => (userId, prod))
//          return userRatings
//        }

        //val ratings = userRatingsFeatures.flatMap(x => getProds(x._1, x._2._1))
        val ratings = userRatingsFeatures.flatMap{
          case (userId, (products, _)) => products.map(prod => (userId, prod))
        }

        val userVectors = scala.collection.mutable.Map[Int, DenseVector[Double]]()
        for (user <- userRatingsFeatures){
          userVectors(user._1) = user._2._2
        }

        val positiveRatingsRepeated = ratings.flatMap(x => Vector.fill(NUM_OF_NEGATIVE_PER_IMPLICIT)(x)).toVector
        val negativeRatings = positiveRatingsRepeated.map(x => (x._1, x._2, Random.nextInt(numProds) + 1))

        //val allRatings = new Vector[(Int, Int, Int)](negativeRatings) // TODO: REMOVE THE NEGATIVE STUFF THAT IS NOT ACTUALLY NEGATIVE SAMPLING?
        val allRatings = negativeRatings
        val sampledRatings = Random.shuffle(allRatings.toList).toVector.slice(0, 20000) // TODO: ADD PROPER CONSTANTS

        for (sampledPoint <- sampledRatings) {
          //gradientSinglePoint(sampledPoint._1, sampledPoint._2, sampledPoint._3, userVectors.apply(sampledPoint._1), prodMat)
          val userId = sampledPoint._1         // TODO: Make this method more beautiful
          val prodPos = sampledPoint._2
          val prodNeg = sampledPoint._3
          val userVector = userVectors.apply(userId)
          var x_uij = userVector.dot(prodMat(prodPos, ::).t) - userVector.dot(prodMat(prodNeg, ::).t)

          var scale = math.exp(-x_uij) / (1 + math.exp(-x_uij))
          prodMat(prodPos, ::) :+= (userVector :* scale).t :* alpha
          prodMat(prodNeg, ::) :+= (-userVector :* scale).t :* alpha

          val newUserVector = userVector + ((prodMat(prodPos, ::) - prodMat(prodNeg, ::)) :* scale).t :* alpha
          userVectors(userId) = newUserVector
        }

        return List((prodMat, userVectors.toArray)).iterator
      }

      def optimizeMF(ratings: RDD[(Int, Int)], rank: Int = 10,
                     numIterations: Int = 10, numPartitions: Int = 4): (DenseMatrix[Double], DenseMatrix[Double]) = {

        val numProds = ratings.map(x => x._2).max()

        // Partition by user: also create the distributed vector
        val ratingsByUser = ratings.groupByKey().persist()
        val userRatingsFeatures = ratingsByUser.map{
          case (userId, products) => (userId, (products, DenseVector.rand[Double](rank)))
        }

        // TODO: In distributed version, partitioning is not important anymore (all products go with user)
        val userPartitioner = new RangePartitioner(numPartitions, userRatingsFeatures)
        var ratingsPartitioned = userRatingsFeatures.partitionBy(userPartitioner).persist()

        //val numUsers = ratingsPartitioned.map(x => x._1).max()

        var prodMat: DenseMatrix[Double] = DenseMatrix.rand[Double](numProds + 1, rank)

        for (i <- 1 until numIterations) {
          val result = ratingsPartitioned.mapPartitions {
            ratings => sampleAndOptimizePartition(ratings, prodMat, numProds)
          }

          prodMat = result.map(x => x._1).reduce((a, b) => a + b) :/ result.count.toDouble

          val userVectorsRDD = result.map(x => x._2).flatMap(x => x.map(y => y))
          ratingsPartitioned = ratingsByUser.join(userVectorsRDD).cache()

          //ratingsPartitioned = resul

          // Average through parameters
          //val num = result.count.toDouble
          //val result2 = result.reduce((a, b) => a + b)

          //prodMat = result2 :/ num
        }

        // Only for evaluation purposes
        val numUsers = ratings.map(x => x._1).max()

        var userMat: DenseMatrix[Double] = DenseMatrix.rand[Double](numUsers + 1, rank)
        val userVectors = ratingsPartitioned.collect().map{ case (userId, (products, vector)) => (userId, vector) } // try without collect
        for (user <- userVectors){
          userMat(user._1,::) := user._2.t
        }

        return (userMat, prodMat)
      }

    }

    val conf = new SparkConf().setAppName("BPR").setMaster("local")
    val sc = new SparkContext(conf)

    val ratingsBPR = sc.textFile("/home/alfredo/Desktop/bpr/training_ratings.txt").map(line => line.split(" ")).map(x => (x(0).toInt, x(1).toInt))
    val (userMat, prodMat) = BPR.optimizeMF(ratingsBPR, 10, 10)
    //val (userMat, prodMat) = DistributedUserBPR.optimizeMF(ratingsBPR, 10, 10)

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
