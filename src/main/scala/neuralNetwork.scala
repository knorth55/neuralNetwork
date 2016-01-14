/*
 * created by knorth55 on 29.12.2015
 */

import java.io.{File,DataInputStream,FileInputStream}
import breeze.linalg._
import breeze.numerics.sigmoid

import scala.collection.mutable.ListBuffer
import scala.util.Random
import Math.{pow,abs}

class neuralNetwork {
  val middle_num: Int = 80                                                      // num of middle layer unit
  val output_num: Int = 10                                                      // num of output layer unit
  val nu: Double = 0.1                                                          // learning rate
  val backPropN: Int = 50                                                       // back propagation step times
  var w1 :DenseMatrix[Double] =  DenseMatrix.zeros(1,1)                         // initialize w1
  var w2 :DenseMatrix[Double] =  DenseMatrix.zeros(1,1)                         // initialize w2
  val current_dir: String = new File(".").getAbsoluteFile().getParent()
  val data_dir = current_dir + "/data"

  def train(): Unit  ={
    val data = load_training_data(data_dir)
    val training_data = data._1
    val training_label = data._2
    val input_num = training_data(1).length                      // num of input layer unit
    w1 = RandomMatrix(input_num+1,middle_num)                    // input layer -> middle layer
    w2 = RandomMatrix(middle_num+1,output_num)                   // middle layer -> output layer
    backPropagation(training_data,training_label)
  }

  def RandomMatrix(cols:Int,rows:Int): DenseMatrix[Double] ={
    val w: DenseMatrix[Double] = DenseMatrix.zeros[Double](cols,rows)
    val r = new Random()
    (0 until cols).foreach {
      case i => {
        (0 until rows).foreach {
          case j => {
            w(i,j) = pow(-1,r.nextInt(2)) * r.nextDouble()
          }
        }
      }
    }
    return w
  }

  def run(img: DenseVector[Double]): (DenseVector[Double],DenseVector[Double],DenseVector[Double]) ={
    // input_output = img
    val input_output_ :DenseVector[Double]= DenseVector.vertcat(img,DenseVector.ones[Double](1))
    val middle_output :DenseVector[Double] = sigmoid(w1.t * input_output_)
    val middle_output_ :DenseVector[Double] = DenseVector.vertcat(middle_output,DenseVector.ones[Double](1))
    val output_output :DenseVector[Double] = sigmoid(w2.t * middle_output_)
    return Tuple3(input_output_,middle_output_,output_output)
  }

  def identify(testing_data: collection.mutable.ListBuffer[DenseVector[Double]],testing_label: collection.mutable.ListBuffer[Int]) : Unit ={
    var total:Int = 0
    var correct:Int = 0
    testing_data.zip(testing_label).foreach {
      case (img,img_label) => {
        var max_value:Double = 0.0
        val run_result: DenseVector[Double] = run(img)._3
        var id :Int = 0
        (0 until output_num).foreach {
          case i => {
            if (max_value < abs(run_result(i))) {
              id = i
              max_value = abs(run_result(i))
            }
          }
        }
        printf("img_label: %d - id: %d\n",img_label,id)
        if (img_label == id) {
          correct += 1
        }
        total += 1
      }
    }
    val accuracy: Double = correct.toDouble / total.toDouble
    printf("correct: %d - total: %d\n",correct,total)
    println(accuracy)
  }

  def backPropagation(training_data: collection.mutable.ListBuffer[DenseVector[Double]],training_label: collection.mutable.ListBuffer[Int]): Unit = {
    val data_test = load_testing_data(data_dir)
    val testing_data = data_test._1
    val testing_label = data_test._2
    (1 to backPropN).foreach {
      case step =>
        training_data.zip(training_label).foreach {
          case (img,img_label) =>
            val output_ref = DenseVector.zeros[Double](output_num)
            output_ref(img_label) = 1.0
            val run_result = run(img)
            val i_o_ = run_result._1
            val m_o_ = run_result._2
            val o_o = run_result._3
            val output_error = (o_o - output_ref) :* sigmoid_d(o_o)
            val middle_error = (w2 * output_error) :* sigmoid_d(m_o_)
            w1 = w1 - nu * (middle_error(0 until middle_error.length-1) * i_o_.t)
            w2 = w2 - nu * (output_error * m_o_.t)
        }
        identify(testing_data,testing_label)
        printf("back propagation step %d finished\n",step)
     }
  }

  def load_training_data(dir: String): (collection.mutable.ListBuffer[DenseVector[Double]],collection.mutable.ListBuffer[Int])= {
    println("start loading training data")
    val training_img_path:String =  dir + "/train-images-idx3-ubyte"
    val training_label_path:String = dir + "/train-labels-idx1-ubyte"
    val training_img_stream = new DataInputStream(new FileInputStream(training_img_path))
    val training_label_stream = new DataInputStream(new FileInputStream(training_label_path))
    val training_data = read_img_stream(training_img_stream)
    val training_label = read_label_stream(training_label_stream)
    println("finish loading training data")
    println(training_label.length)
    return Tuple2(training_data,training_label)
  }

  def load_testing_data(dir: String): (collection.mutable.ListBuffer[DenseVector[Double]],collection.mutable.ListBuffer[Int])= {
    println("start loading testing data")
    val testing_img_path:String =  dir + "/t10k-images-idx3-ubyte"
    val testing_label_path:String = dir + "/t10k-labels-idx1-ubyte"
    val testing_img_stream = new DataInputStream(new FileInputStream(testing_img_path))
    val testing_label_stream = new DataInputStream(new FileInputStream(testing_label_path))
    val testing_data = read_img_stream(testing_img_stream)
    val testing_label = read_label_stream(testing_label_stream)
    println("finish loading testing data")
    println(testing_label)
    return Tuple2(testing_data,testing_label)
  }

  def read_img_stream(img_stream: DataInputStream): collection.mutable.ListBuffer[DenseVector[Double]] = {
    if (img_stream.readInt() != 2051) {
      println("Wrong magic number, expected 2052")
    }
    val count = img_stream.readInt()
    val height = img_stream.readInt()
    val width = img_stream.readInt()
    var list: collection.mutable.ListBuffer[DenseVector[Double]] = new ListBuffer[DenseVector[Double]]();
     (0 until count).foreach{
       case index => {
        var vec:DenseVector[Double] = DenseVector.zeros[Double](height*width)
        (0 until height).foreach {
          case y =>
            (0 until width).foreach{
              case x =>
                vec(y*width+x) = img_stream.readUnsignedByte() / 255.0
            }
        }
        list += vec;
      }
    }
    return list
  }

  def read_label_stream(label_stream: DataInputStream): collection.mutable.ListBuffer[Int] ={
    if (label_stream.readInt() != 2049) {
      println("Wrong magic number, expected 2049")
    }
    val count = label_stream.readInt()
    var list: collection.mutable.ListBuffer[Int] = new ListBuffer[Int]()
    (0 until count).foreach {
      case index => {
        list += label_stream.readByte()
      }
    }
    return list
  }

  def sigmoid_d(x:DenseVector[Double]): DenseVector[Double] ={
    (0 until x.length).foreach {
      case i => {
        x(i) = x(i) * (1 -x(i))
      }
    }
    return x
  }
}

object neuralNetwork {
  def main(args: Array[String]): Unit = {
    val NN = new neuralNetwork()
    NN.train()
  }
}
