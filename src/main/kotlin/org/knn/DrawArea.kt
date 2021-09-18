package org.knn

import java.awt.*
import java.awt.event.MouseAdapter
import java.awt.event.MouseEvent
import java.awt.event.MouseMotionAdapter
import javax.swing.JComponent

class DrawArea : JComponent() {
    // Image in which we're going to draw
    private var image: Image? = null

    // Graphics2D object ==> used to draw on
    private var g2: Graphics2D? = null

    // Mouse coordinates
    private var currentX = 0
    private var currentY = 0
    private var oldX = 0
    private var oldY = 0
    override fun paintComponent(g: Graphics) {
        if (image == null) {
            // image to draw null ==> we create
            val graphic = createImage(size.width, size.height)
            image = graphic
            g2 = graphic.graphics as Graphics2D
            // enable antialiasing
            g2!!.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            // clear draw area
            clear()
        }
        g.drawImage(image, 0, 0, null)
    }

    // now we create exposed methods
    fun clear() {
        g2!!.paint = Color.white
        // draw white on entire draw area to clear
        g2!!.fillRect(0, 0, size.width, size.height)
        g2!!.paint = Color.black
        repaint()
    }

    fun red() {
        // apply red color on g2 context
        g2!!.paint = Color.red
    }

    fun black() {
        g2!!.paint = Color.black
    }

    fun magenta() {
        g2!!.paint = Color.magenta
    }

    fun green() {
        g2!!.paint = Color.green
    }

    fun blue() {
        g2!!.paint = Color.blue
    }

    init {
        isDoubleBuffered = false
        addMouseListener(object : MouseAdapter() {
            override fun mousePressed(e: MouseEvent) {
                // save coord x,y when mouse is pressed
                oldX = e.getX()
                oldY = e.getY()
            }
        })
        addMouseMotionListener(object : MouseMotionAdapter() {
            override fun mouseDragged(e: MouseEvent) {
                // coord x,y when drag mouse
                currentX = e.getX()
                currentY = e.getY()
                if (g2 != null) {
                    // draw line if g2 context not null
                    g2!!.drawLine(oldX, oldY, currentX, currentY)
                    // refresh draw area to repaint
                    repaint()
                    // store current coords x,y as olds x,y
                    oldX = currentX
                    oldY = currentY
                }
            }
        })
    }
}