package org.knn

import java.awt.*
import java.awt.image.BufferedImage
import java.io.DataInputStream
import java.io.FileInputStream
import java.util.concurrent.Semaphore
import javax.swing.JFrame
import javax.swing.SwingUtilities
import kotlin.math.pow
import kotlin.random.Random

import javax.swing.JButton

import javax.swing.JPanel

import java.awt.event.*

import javax.swing.JComponent



const val TRAIN_TEST_IMG_PATH = "src/main/resources/dataset/train-images-idx3-ubyte"
const val TRAIN_TEST_LABEL_PATH = "src/main/resources/dataset/train-labels-idx1-ubyte"
const val ALPHA = 0.2F

data class ImageNN(
    val width: Int,
    val height: Int,
    val data: IntArray
) {
    fun toBufferedImage(): BufferedImage {
        fun genRgb(grayScale: Int): Int = (255 - grayScale).let {
            Color(it, it, it).rgb
        }

        val image = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
        for (h in 0 until height) {
            for (w in 0 until width) {
                image.setRGB(w, h, genRgb(data[h * width + w]))
            }
        }
        return image
    }
}

fun showImage(image: BufferedImage, width: Int = 300, height: Int = 300) {
    val canvas = object : Canvas() {
        override fun paint(g: Graphics) {
            g.drawImage(image, 0, 0, width, height, null)
        }
    }.apply {
        size = Dimension(width, height)
        background = Color.BLACK
    }

    val win = JFrame().apply {
        add(canvas)
        pack()
        defaultCloseOperation = JFrame.DISPOSE_ON_CLOSE
    }

    SwingUtilities.invokeAndWait {
        win.isVisible = true
    }

    val sem = Semaphore(1)

    win.addWindowListener(object : WindowAdapter() {
        override fun windowClosed(e: WindowEvent?) {
            sem.release()
        }
    })

    sem.acquire()
}

fun getImage(): ImageNN {
    var img: ImageNN

    FileInputStream(TRAIN_TEST_IMG_PATH).buffered().use { input ->
        val dis = DataInputStream(input)
        dis.readInt().run {
            if (this != 0x00000803) {
                throw Exception("Magic number expected")
            }
        }

        val imageNumber = dis.readInt()

        val imageHeight = dis.readInt()
        val imageWidth = dis.readInt()

        print("$imageNumber $imageHeight $imageWidth")

        val data = IntArray(imageWidth * imageHeight)
        for (h in 0 until imageHeight) {
            for (w in 0 until imageWidth) {
                data[h * imageWidth + w] = dis.readUnsignedByte()
            }
        }

        img = ImageNN(imageWidth, imageHeight, data)
    }

    return img
}

fun getLabels(path: String): List<Byte> {
    FileInputStream(path).buffered().use { inputStream ->
        val dis = DataInputStream(inputStream)
        dis.readInt().run {
            if (this != 0x00000801) {
                throw Exception("Magic number expected for file $path")
            }
        }
        val labelNumber = dis.readInt()

        return ArrayList<Byte>().apply {
            for (i in 1..labelNumber) {
                add(dis.readUnsignedByte().toByte())
            }
        }
    }
}

fun List<Float>.dot(x: List<Float> ): Float {
    if (this.size != x.size) {
        throw Exception("Vectors are not same size")
    }
    this.size.let {
        var stack = 0F
        for (n in 0 until it) {
           stack +=  x[n] * this[n]
        }
        return stack
    }
}

fun List<List<Float>>.T(): List<List<Float>> {
    val col = this.size
    val row = this[0].size
    val transposedMatrix = Array(row) { FloatArray(col).toMutableList() }.toMutableList()
    for (rowIndex in 0 until row) {
        for (colIndex in 0 until col) {
            transposedMatrix[rowIndex][colIndex] = this[colIndex][rowIndex]
        }
    }

    return transposedMatrix
}

@JvmName("TFloat")
fun List<Float>.T(): List<List<Float>> = this.fold(mutableListOf()) { acc, x ->
    acc.add(mutableListOf(x))
    acc
}

fun List<Float>.ewSum(y: List<Float>): List<Float> = this.zip(y){ a, b -> a + b }
fun List<Float>.ewSub(y: List<Float>): List<Float> = this.zip(y){ a, b -> a - b }
fun List<Float>.scalarProduct(x: Float): List<Float> = this.map { it * x}

fun List<Float>.matrixProduct(matrix: List<List<Float>>): List<Float> {

    if (this.size != matrix.size) {
        throw Exception("Sizes does not match")
    }

    return matrix.T().map {
        it.dot(this)
    }
}

@JvmName("matrixProductFloat")
fun List<List<Float>>.matrixProduct(matrix: List<List<Float>>): List<List<Float>> = this.map { it.matrixProduct(matrix) }

fun List<Float>.relu(): List<Float> = this.map {
    when {
        it > 0 -> it
        else -> 0F
    }
}

@JvmName("reluFloat")
fun List<List<Float>>.relu(): List<List<Float>> = this.map { it.relu() }

fun List<List<Float>>.toVector(): List<Float> {
    if (this.size > 1) {
        throw Exception("Matrix size greater than 1, can not convert to Vector")
    }

    return this[0]
}

fun List<Float>.toMatrix(): List<List<Float>> = mutableListOf(this)

fun randomVector(size: Int): List<Float> = IntRange(1, size).fold(mutableListOf()){acc, _ ->
    acc.add(Random.nextDouble(-1.0, 1.0).toFloat())
    acc
}


fun randomMatrix(colSize: Int, rowSize: Int): List<List<Float>> = IntRange(1, colSize).fold(mutableListOf()){acc, _ ->
    acc.add(randomVector(rowSize))
    acc
}

class DataNN(
    val alpha: Float,
    val weights: List<Float>,
    val inputs: List<Float>,
    val output: Float,
    val tries:Int = 400
)

fun manyToOneNN(data: DataNN): List<Float> {
    val alpha = data.alpha
    var weights = data.weights
    val inputs = data.inputs
    val goal = data.output
    var error = 1.0
    var tries = 0

    while (error >= 0.0001F && tries < data.tries) {

        val prediction = inputs.dot(weights)
        val delta =  prediction - goal
        error = delta.pow(2).toDouble()
        val weightDeltas = inputs.map { x -> delta * x}
        weights = weightDeltas.zip(weights).map { pair -> pair.second - (pair.first * alpha) }.toMutableList()
        println("Error: $error - Weight: ${weights.joinToString(separator = ", ")} - Tries: $tries")
        tries++
    }

    return weights
}

fun manyToManyNN() {
    val alpha = 0.004F
    val inputs = mutableListOf(0.1F, -0.2F, 0.4F)
    val goals = mutableListOf(1F, 3F, -1F) //Also output

    val weights_matrix = randomMatrix(inputs.size, goals.size).toMutableList()

    for (row in 0 until inputs.size ) {
        println("Row index $row")
        weights_matrix[row] = manyToOneNN(DataNN(
            alpha = alpha,
            weights = weights_matrix[row],
            inputs = inputs,
            output = goals[row]
        ))
    }

    println("Result")
    println(weights_matrix.map { it.dot(inputs) }.joinToString(separator = ", "))
}

abstract class Layer(var values: List<List<Float>>, var weights: List<List<Float>>) {

    abstract fun forward(): List<List<Float>>
    abstract fun back(deltas: List<List<Float>> = emptyList()):List<List<Float>>
}

class InputLayer(values: List<List<Float>>, weights: List<List<Float>>): Layer(values, weights) {

    override fun forward(): List<List<Float>> = values.matrixProduct(weights).relu()

    override fun back(deltas: List<List<Float>>): List<List<Float>> {
        this.weights = weights.zip(values.T().map { it.matrixProduct(deltas) }.map { it.scalarProduct(ALPHA) })
            .map { x -> x.first.ewSum(x.second) }
        return emptyList()
    }
}

class HiddenLayer(values: List<List<Float>>, weights: List<List<Float>>): Layer(values, weights) {
    override fun forward(): List<List<Float>> = values.matrixProduct(weights.relu())
    override fun back(deltas: List<List<Float>>): List<List<Float>> {
        val newDelta = deltas.matrixProduct(weights.T()).zip(values).let { listOfPairs -> listOfPairs.map { pair -> pair.first.zip(pair.second).map { if (it.second > 0) it.first else 0 } } } as List<List<Float>>
        this.weights = weights.zip(values.T().map { it.matrixProduct(deltas) }.map { it.scalarProduct(ALPHA) })
            .map { x -> x.first.ewSum(x.second) }
        return newDelta
    }
}

class OutputLayer(values: List<List<Float>>, weights: List<List<Float>>, var expectedValues: List<Float>): Layer(values, weights) {
    override fun forward(): List<List<Float>> {
        return emptyList()
    }

    override fun back(deltas: List<List<Float>>): List<List<Float>> {
        return expectedValues.toMatrix().zip(values).map {it.first.ewSub(it.second) }
    }

    fun getError(): Float {
        return values.zip(expectedValues.toMatrix()){x, y -> x.ewSub(y)}.toVector().map{it.pow(2)}.reduce{a, b -> a + b}
    }
}

//data class Connexion(val from: Neuron, val to: Neuron, val weight: Float)

//data class Neuron(val value: Float, val connexions: List<Connexion>)

fun NN() {


    val output = mutableListOf(1F, 1F, 0F, 0F, 1F).T()

    val layer0 = mutableListOf(
        mutableListOf( 1F, 0F, 1F),
        mutableListOf( 0F, 1F, 1F),
        mutableListOf( 0F, 0F, 1F),
        mutableListOf( 1F, 1F, 1F),
        mutableListOf( 1F, 1F, 0F)
    )

    var weights0to1: List<List<Float>>  = mutableListOf(
        mutableListOf(-0.16595599F,  0.44064899F, -0.99977125F, -0.39533485F),
        mutableListOf(-0.70648822F, -0.81532281F, -0.62747958F, -0.30887855F),
        mutableListOf(-0.20646505F,  0.07763347F, -0.16161097F,  0.370439F),
    )

    var weights1to2: List<List<Float>> = mutableListOf(
        mutableListOf(-0.5910955F),
        mutableListOf(0.75623487F),
        mutableListOf(-0.94522481F),
        mutableListOf( 0.34093502F),
    )

    val inputLayer = InputLayer(emptyList(), weights0to1)
    val hiddenLayer = HiddenLayer(emptyList(), weights1to2)
    val outputLayer = OutputLayer(emptyList(), emptyList(), emptyList())

    var counter = 0
    var layer2Error = 1F

    val layers = listOf(inputLayer, hiddenLayer, outputLayer)
    while (counter < 61 ) {

        for (i in 0 until layer0.size) {
            inputLayer.values = layer0[i].toMatrix()
            outputLayer.expectedValues = output[i]

            layers.reduce{acc, layer ->
                layer.values = acc.forward()
                layer
            }

            layer2Error = outputLayer.getError()

            // Get delta last layer
            layers.foldRight(emptyList<List<Float>>()){ acc, layer -> acc.back(layer)}

        }

        println("$layer2Error - $counter")
        counter++
    }

    inputLayer.values = mutableListOf(mutableListOf( 1F, 1F, 0F))

    layers.reduce{acc, layer ->
        layer.values = acc.forward()
        layer
    }

    println(outputLayer.values)
}

fun main() {
   NN()
}