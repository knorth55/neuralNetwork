/*
 * created by shingo kitagawa on 29.12.2015
 */

import java.io.{File,DataInputStream,FileInputStream}
import breeze.linalg._
import breeze.numerics.sigmoid


class neuralNetwork {
  val input_num: Int = 100;                                         // num of input layer unit
  val middle_num: Int = 10;                                         // num of middle layer unit
  val output_num: Int = 10 ;                                        // num of output layer unit
  val nu: Double = 0.1                                              // learning rate
  val backPropN: Int = 50;                                          // back propagation step times
  var w1: DenseMatrix[Double] = DenseMatrix.rand(input_num+1,middle_num);       // input layer -> middle layer
  var w2: DenseMatrix[Double] = DenseMatrix.rand(middle_num+1,output_num);      // middle layer -> output layer
  val current_dir: String = new File(".").getAbsoluteFile().getParent();
  val data_dir = current_dir + "/data";

  def load_data(data_dir:String): (collection.mutable.ListBuffer[DenseVector[Double]],collection.mutable.ListBuffer[Int],collection.mutable.ListBuffer[DenseVector[Double]],collection.mutable.ListBuffer[Int])=
  {
    val training_data: collection.mutable.ListBuffer[DenseVector[Double]] = load_training_data(data_dir)._1;
    val training_label: collection.mutable.ListBuffer[Int] = load_training_data(data_dir)._2;
    val testing_data: collection.mutable.ListBuffer[DenseVector[Double]] = load_testing_data(data_dir)._1;
    val testing_label: collection.mutable.ListBuffer[Int] = load_testing_data(data_dir)._2;
    return Tuple4(training_data,training_label,testing_data,testing_label)
  }

  def run(img: DenseVector[Double]): (DenseVector[Double],DenseVector[Double],DenseVector[Double]) ={
    // input_output = img
    val input_output_ :DenseVector[Double]= DenseVector.vertcat(img,DenseVector.ones[Double](1));
    val middle_output :DenseVector[Double] = (w1 * input_output_.t).toDenseVector.map(x => sigmoid(x));
    val middle_output_ :DenseVector[Double] = DenseVector.vertcat(middle_output,DenseVector.ones[Double](1));
    val output_output :DenseVector[Double] = (w2 * middle_output_.t).toDenseVector.map(x => sigmoid(x));
    return Tuple3(input_output_,middle_output_,output_output)
  }

  def backPropagation(training_data: collection.mutable.ListBuffer[DenseVector[Double]],training_label: collection.mutable.ListBuffer[Int]): Unit = {
    (1 to backPropN).foreach {
      case step =>
        training_data.zip(training_label).foreach {
          case (img,img_label) =>
            var output_ref = DenseVector.zeros[Double](output_num);
            output_ref(img_label) = 1.0;
            val i_o_ = run(img)._1;
            val m_o_ = run(img)._2;
            val o_o = run(img)._3;
            val output_error = (o_o - output_ref) :* (o_o).map {case x => sigmoid_d(x)};
            val middle_error_ = ((w2 * output_error).toDenseVector) :* (m_o_).map {case x => sigmoid_d(x)};
        }
    }
  }

  def load_training_data(data_dir: String): (collection.mutable.ListBuffer[DenseVector[Double]],collection.mutable.ListBuffer[Int])= {
    val training_img_path:String =  data_dir + "/train-images-idx3-ubyte";
    val training_label_path:String = data_dir + "/train-labels-idx1-ubyte";
    val training_img_stream = new DataInputStream(new FileInputStream(training_img_path));
    val training_label_stream = new DataInputStream(new FileInputStream(training_label_path));
    val training_data = read_img_stream(training_img_stream)
    val training_label = read_label_stram(training_label_stream)
    return Tuple2(training_data,training_label)
  }

  def load_testing_data(data_dir: String): (collection.mutable.ListBuffer[DenseVector[Double]],collection.mutable.ListBuffer[Int])= {
    val testing_img_path:String =  data_dir + "/t10k-images-idx3-ubyte";
    val testing_label_path:String = data_dir + "/t10k-labels-idx1-ubyte";
    val testing_img_stream = new DataInputStream(new FileInputStream(testing_img_path));
    val testing_label_stream = new DataInputStream(new FileInputStream(testing_label_path));
    val testing_data = read_img_stream(testing_img_stream)
    val testing_label = read_label_stram(testing_label_stream)
    return Tuple2(testing_data,testing_label)

  }

  def read_img_stream(img_stream: DataInputStream): collection.mutable.ListBuffer[DenseVector[Double]] = {
    if (img_stream.readInt() != 2051) {
      println("Wrong magic number, expected 2052");
    }
    val count = img_stream.readInt();
    val height = img_stream.readInt();
    val width = img_stream.readInt();
    var returnlist: collection.mutable.ListBuffer[DenseVector[Double]] = new collection.mutable.ListBuffer[DenseVector[Double]]();
     (0 until count).foreach{
       case index => {
        var vec:DenseVector[Double] = DenseVector.zeros[Double](height*width);
        (0 until height).foreach {
          case y =>
            (0 until width).foreach{
              case x =>
                vec(y*width+x) = img_stream.readUnsignedByte();
            }
        }
        returnlist += vec;
      }
    }
    return returnlist
  }

  def read_label_stream(label_stream: DataInputStream): collection.mutable.ListBuffer[Int] ={

  }

  def sigmoid_d(x:Double): Double ={
    return (1-x) * x;
  }
}

object neuralNetwork {
  def main(args: Array[String]): Unit = {
    var NN = new neuralNetwork();
  }
}
