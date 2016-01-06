/*
 * created by shingo kitagawa on 29.12.2015
 */

import breeze.linalg._
import breeze.numerics._

class neuralNetwork {
  def initialize(flag_training: Int = 1): Unit = {
    val input_num: Int = 100;                                         // num of input layer unit
    val middle_num: Int = 10;                                         // num of middle layer unit
    val output_num: Int = 10 ;                                        // num of output layer unit
    val nu: Double = 0.1                                              // learning rate
    val backPropN: Int = 50;                                          // back propagation step times
    var w1 = DenseMatrix.rand(input_num+1,middle_num);       // input layer -> middle layer
    var w2 = DenseMatrix.rand(middle_num+1,output_num);      // middle layer -> output layer
    if (flag_training == 1) {
      val training_data: collection.immutable.List[breeze.linalg.DenseVector[Double]] = load_training_data()._1;
      val training_label: collection.immutable.List[Int] = load_training_data()._2;
    }
    val testing_data: collection.immutable.List[breeze.linalg.DenseVector[Double]] = load_testing_data()._1;
    val testing_label: collection.immutable.List[Int] = load_testing_data()._2;
  }

  def run(img: breeze.linalg.DenseVector[Double]): (breeze.linalg.DenseVector[Double],breeze.linalg.DenseVector[Double],breeze.linalg.DenseVector[Double]) ={
    // input_output = img
    val input_output_  = img :+ 1.0;
    val middle_output  = (w1 * input_output_.t).map { case x => sigmoid(x)};
    val middle_output_  = middle_output :+ 1.0;
    val output_output = (w2 * middle_output_.t).map { case x => sigmoid(x)};
    return Tuple3(input_output_,middle_output_,output_output)
  }

  def backPropagation(training_data: collection.immutable.List[breeze.linalg.DenseVector[Double]],training_label: collection.immutable.List[Int]): Unit = {
    (1 to backPropN).foreach {
      case step =>
        zip(training_data,training_label).foreach {
          case (img,img_label) =>
            var output_ref = breeze.linalg.zeros(output_num);
            output_ref(img_label) = 1.0;
            val i_o_ = run(img)._1;
            val m_o_ = run(img)._2;
            val o_o = run(img)._3;
            val output_error = (o_o - output_ref) :* (o_o).map {case x => sigmoid_d(x)}
            val middle_error_ = (w2 * output_error) :* (m_o_).map {case x => sigmoid_d(x)}            
        }
    }
  }

  def load_training_data(): (collection.immutable.List[breeze.linalg.DenseVector[Double]],collection.immutable.List[Int])= {
  }

  def load_testing_data(): (collection.immutable.List[breeze.linalg.DenseVector[Double]],collection.immutable.List[Int])= {
  }

  def sigmoid_d(x:Double): Double ={
    return (1-x) * x;
  }
}

object neuralNetwork {
  def main(args: Array[String]): Unit = {
    var NN = new neuralNetwork()
  }
}
