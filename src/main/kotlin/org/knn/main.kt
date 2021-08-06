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
import java.util.stream.IntStream
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

fun List<Float>.sumProduct(matrix: List<List<Float>>): List<Float> {

    val list = mutableListOf<Float>()

    matrix.forEach {
        if (this.size != it.size) {
            throw Exception("Size does not match")
        }
    }

    for (col in matrix.indices) {
        var agg = 0F
        for (row in matrix[col].indices) {
            agg += matrix[row][col] * this[row]
        }
        list.add(agg)
    }

    return list
}

fun randomVector(size: Int): List<Float> = IntRange(1, size).fold(mutableListOf()){acc, _ ->
    acc.add(Random.nextDouble(-1.0, 1.0).toFloat())
    acc
}


fun randomMatrix(rowSize: Int, colSize: Int): List<List<Float>> = IntRange(1, colSize).fold(mutableListOf()){acc, _ ->
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

fun main() {

    println(mutableListOf(2F,1F,3F).sumProduct(
        mutableListOf(
            mutableListOf(1F,4F,7F),
            mutableListOf(2F,5F,8F),
            mutableListOf(3F,6F,9F)
        )
    ))

    manyToOneNN(
        DataNN(
            alpha = 0.0007F,
            weights = randomVector(3),
            inputs = mutableListOf(0.1F, -0.2F, 0.4F),
            output = -1F
    ))
}