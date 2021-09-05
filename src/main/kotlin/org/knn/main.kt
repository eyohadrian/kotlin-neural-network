package org.knn

import java.awt.Canvas
import java.awt.Color
import java.awt.Dimension
import java.awt.Graphics
import java.awt.event.WindowAdapter
import java.awt.event.WindowEvent
import java.awt.image.BufferedImage
import java.io.DataInputStream
import java.io.FileInputStream
import java.util.concurrent.Semaphore
import javax.swing.JFrame
import javax.swing.SwingUtilities
import kotlin.math.pow
import kotlin.random.Random

const val TRAIN_TEST_IMG_PATH = "src/main/resources/dataset/train-images-idx3-ubyte"

data class Image(
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

fun getImage(): Image {
    var img: Image

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

        img = Image(imageWidth, imageHeight, data)
    }

    return img
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

fun getError(lastLayer: List<List<Float>>, expectedOutput: List<List<Float>> ): Float {
    return lastLayer.zip(expectedOutput){x, y -> x.ewSub(y)}.toVector().map{it.pow(2)}.reduce{a, b -> a + b}
}

fun correctWeights(layer: List<List<Float>>, deltas: List<List<Float>>, weights: List<List<Float>>, alpha: Float = 0.2F): List<List<Float>> =
    weights.zip(layer.T().map { it.matrixProduct(deltas) }.map { it.scalarProduct(alpha) }).map { x -> x.first.ewSum(x.second) }

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


fun main() {

    val alpha = 0.2F
    val output = mutableListOf(mutableListOf(1F))

    val layer0 = mutableListOf(mutableListOf( 1F, 0F, 1F))

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

    var counter = 0
    var layer2Error = 1F

    while (counter < 61 ) {
        val layer1 = layer0.matrixProduct(weights0to1).relu()

        val layer2 = layer1.matrixProduct(weights1to2)
        layer2Error = getError(layer2, output)

        // Get delta last layer
        val layer2Delta = output.zip(layer2).map {it.first.ewSub(it.second) }
        val layer1Delta: List<List<Float>> =
            layer2Delta.matrixProduct(weights1to2.T()).zip(layer1).let { listOfPairs -> listOfPairs.map { pair -> pair.first.zip(pair.second).map { if (it.second > 0) it.first else 0 } } } as List<List<Float>>

        weights1to2 = correctWeights(layer1, layer2Delta, weights1to2, alpha)
        weights0to1 = correctWeights(layer0, layer1Delta, weights0to1, alpha)

        println("$layer2Error - $counter")
        counter++
    }
}