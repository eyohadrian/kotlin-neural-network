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
        return image;
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

fun main() {

    val alpha = 0.7
    var weight = 0.1
    val input = 1
    val goal = 14

    IntStream.range(0, 400).forEach{
        val prediction = input * weight
        val delta =  prediction - goal
        val error = delta.pow(2)
        val weight_delta = delta * input
        weight -= weight_delta * alpha

        println("Error: $error - Weight: $weight")
    }
}