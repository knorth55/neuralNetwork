/*
 * created by knorth55 on 15.01.16
 */

import java.io.{DataInputStream, File, FileInputStream}
import breeze.linalg._
import scala.collection.mutable.ListBuffer

class Mnist {
  val current_dir: String = new File(".").getAbsoluteFile.getParent
  val data_dir = current_dir + "/data"

  protected[this] def load_training_data(dir: String): (ListBuffer[DenseVector[Double]],ListBuffer[Int])= {
    println("start loading training data")
    val training_img_path:String =  dir + "/train-images-idx3-ubyte"
    val training_label_path:String = dir + "/train-labels-idx1-ubyte"
    val training_img_stream = new DataInputStream(new FileInputStream(training_img_path))
    val training_label_stream = new DataInputStream(new FileInputStream(training_label_path))
    val training_data = read_img_stream(training_img_stream)
    val training_label = read_label_stream(training_label_stream)
    println("finish loading training data")
    Tuple2(training_data,training_label)
  }

  protected[this] def load_testing_data(dir: String): (ListBuffer[DenseVector[Double]],ListBuffer[Int])= {
    println("start loading testing data")
    val testing_img_path:String =  dir + "/t10k-images-idx3-ubyte"
    val testing_label_path:String = dir + "/t10k-labels-idx1-ubyte"
    val testing_img_stream = new DataInputStream(new FileInputStream(testing_img_path))
    val testing_label_stream = new DataInputStream(new FileInputStream(testing_label_path))
    val testing_data = read_img_stream(testing_img_stream)
    val testing_label = read_label_stream(testing_label_stream)
    println("finish loading testing data")
    Tuple2(testing_data,testing_label)
  }

  private[this] def read_img_stream(img_stream: DataInputStream): ListBuffer[DenseVector[Double]] = {
    if (img_stream.readInt != 2051) {
      println("Wrong magic number, expected 2052")
    }
    val count = img_stream.readInt
    val height = img_stream.readInt
    val width = img_stream.readInt
    var list: ListBuffer[DenseVector[Double]] = new ListBuffer[DenseVector[Double]]
     (0 until count).foreach{
       case index =>
        var vec:DenseVector[Double] = DenseVector.zeros[Double](width*height)
        (0 until height).foreach {
          case y =>
            (0 until width).foreach{
              case x =>
                vec(y*width+x) = img_stream.readUnsignedByte / 255.0
            }
        }
        list += vec;
    }
    list
  }

  private[this] def read_label_stream(label_stream: DataInputStream): ListBuffer[Int] ={
    if (label_stream.readInt != 2049) {
      println("Wrong magic number, expected 2049")
    }
    val count = label_stream.readInt
    var list: ListBuffer[Int] = new ListBuffer[Int]
    (0 until count).foreach {
      case index =>
        list += label_stream.readByte
    }
    list
  }

}

