/*
 * created by shingo kitagawa on 29.12.2015
 */

import math.{E,pow}
import org.opencv.core.Mat
import org.opencv.highgui.Highgui

object neuralNetwork {
  def main(args:Array[String]): Unit ={
    var middle: Int = 10;
    // num of middle layer
    var w1: collection.mutable.ArrayBuffer[collection.mutable.ArrayBuffer[Double]] = collection.mutable.ArrayBuffer.fill(middle, 100+1)(1.0);
    //input layer -> middle layer
    var w2: collection.mutable.ArrayBuffer[collection.mutable.ArrayBuffer[Double]] = collection.mutable.ArrayBuffer.fill(5, middle+1)(1.0);
    //middle layer -> output layer
    var beta: Double = 1.0;
  }

  def backPropagation(w1: collection.mutable.ArrayBuffer[collection.mutable.ArrayBuffer[Double]], w2:collection.mutable.ArrayBuffer[collection.mutable.ArrayBuffer[Double]]):
      collection.mutable.ListBuffer[collection.mutable.ArrayBuffer[collection.mutable.ArrayBuffer[Double]]]= {
    val returnList = new collection.mutable.ListBuffer[collection.mutable.ArrayBuffer[collection.mutable.ArrayBuffer[Double]]]();
    return returnList;
  }

  def sigmoid(input: collection.mutable.ArrayBuffer[Double],beta:Double): Double = {
    val sumUp: Double = input.sum;
    return 1.0/(1.0+ pow(math.E,-sumUp*beta));
  }
}


