/*
 * created by knorth55 on 29.12.2015
 */

import breeze.linalg._
import breeze.numerics.sigmoid

import scala.collection.mutable.ListBuffer
import scala.util.Random
import Math.{pow,abs}

class NeuralNetwork extends Mnist{
  val middle_num: Int = 80                                                      // num of middle layer unit
  val output_num: Int = 10                                                      // num of output layer unit
  val nu: Double = 0.1                                                          // learning rate
  val backPropN: Int = 100                                                      // back propagation step times
  var w1 :DenseMatrix[Double] =  DenseMatrix.zeros(1,1)                         // initialize w1
  var w2 :DenseMatrix[Double] =  DenseMatrix.zeros(1,1)                         // initialize w2

  def train(): Unit  ={
    val data = load_training_data(data_dir)
    val training_data = data._1
    val training_label = data._2
    val input_num = training_data.head.length                      // num of input layer unit
    w1 = randomMatrix(input_num+1,middle_num)                    // input layer -> middle layer
    w2 = randomMatrix(middle_num+1,output_num)                   // middle layer -> output layer
    backPropagation(training_data,training_label)
  }

  private[this] def randomMatrix(rows:Int,cols:Int): DenseMatrix[Double] ={
    val w: DenseMatrix[Double] = DenseMatrix.zeros[Double](rows,cols)
    val r = new Random
    (0 until rows).foreach {
      case i =>
        (0 until cols).foreach {
          case j =>
            w(i,j) = pow(-1,r.nextInt(2)) * r.nextDouble
        }
    }
    w
  }

  def run(img: DenseVector[Double]): (DenseVector[Double],DenseVector[Double],DenseVector[Double])  ={
    // input_output = img
    val input_output_ :DenseVector[Double] = DenseVector.vertcat(img,DenseVector.ones[Double](1))
    val middle_output :DenseVector[Double] = sigmoid(w1.t * input_output_)
    val middle_output_ :DenseVector[Double] = DenseVector.vertcat(middle_output,DenseVector.ones[Double](1))
    val output_output :DenseVector[Double] = sigmoid(w2.t * middle_output_)
    Tuple3(input_output_,middle_output_,output_output)
  }

  def identify(testing_data: ListBuffer[DenseVector[Double]],testing_label: ListBuffer[Int]) : Unit ={
    var total:Int = 0
    var correct:Int = 0
    testing_data.zip(testing_label).foreach {
      case (img,img_label) =>
        var max_value:Double = 0.0
        val data = run(img)
        val o_o = data._3
        var id :Int = 0
        (0 until output_num).foreach {
          case i =>
            if (max_value < abs(o_o(i))) {
              id = i
              max_value = abs(o_o(i))
            }
        }
        println(s"img_label: $img_label - id: $id")
        if (img_label == id) {
          correct += 1
        }
        total += 1
    }
    val accuracy: Double = correct.toDouble / total.toDouble
    println(s"correct: $correct - total: $total")
    println(s"accuracy: $accuracy")
  }

  def backPropagation(training_data: ListBuffer[DenseVector[Double]],training_label: ListBuffer[Int]): Unit = {
    val data_test = load_testing_data(data_dir)
    val testing_data = data_test._1
    val testing_label = data_test._2
    (1 to backPropN).foreach {
      case step =>
        training_data.zip(training_label).foreach {
          case (img,img_label) =>
            val output_ref = DenseVector.zeros[Double](output_num)
            output_ref(img_label) = 1.0
            val data = run(img)
            val o_o = data._3
            val m_o_ = data._2
            val i_o_ = data._1
            val output_error = (o_o - output_ref) :* sigmoid_d(o_o)
            val middle_error = (w2 * output_error) :* sigmoid_d(m_o_)
            w2 -= nu * (output_error * m_o_.t)
            w1 -= nu * (middle_error(0 until middle_error.length-1) * i_o_.t)
        }
        identify(testing_data,testing_label)
        println(s"back propagation step $step finished")
     }
  }

  private[this] def sigmoid_d(x:DenseVector[Double]): DenseVector[Double] ={
    (0 until x.length).foreach {
      case i =>
        x(i) = x(i) * (1 -x(i))
    }
    x
  }
}

object neuralNetwork {
  def main(args: Array[String]): Unit = {
    val NN = new NeuralNetwork()
    NN.train()
  }
}
