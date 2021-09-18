package org.knn

import java.awt.BorderLayout
import java.awt.Container
import java.awt.event.ActionListener
import javax.swing.JButton
import javax.swing.JFrame
import javax.swing.JPanel

class SwingPaint {
    var clearBtn: JButton? = null
    var blackBtn: JButton? = null
    var blueBtn: JButton? = null
    var greenBtn: JButton? = null
    var redBtn: JButton? = null
    var magentaBtn: JButton? = null
    var drawArea: DrawArea? = null
    var actionListener = ActionListener { e ->
        if (e.source === clearBtn) {
            drawArea?.clear()
        } else if (e.source === blackBtn) {
            drawArea?.black()
        } else if (e.source === blueBtn) {
            drawArea?.blue()
        } else if (e.source === greenBtn) {
            drawArea?.green()
        } else if (e.source === redBtn) {
            drawArea?.red()
        } else if (e.source === magentaBtn) {
            drawArea?.magenta()
        }
    }

    fun show() {
        // create main frame
        val frame = JFrame("Swing Paint")
        val content: Container = frame.contentPane
        // set layout on content pane
        content.layout = BorderLayout()
        // create draw area
        drawArea = DrawArea()

        // add to content pane
        content.add(drawArea, BorderLayout.CENTER)

        // create controls to apply colors and call clear feature
        val controls = JPanel()
        clearBtn = JButton("Clear")
        clearBtn!!.addActionListener(actionListener)
        blackBtn = JButton("Black")
        blackBtn!!.addActionListener(actionListener)
        blueBtn = JButton("Blue")
        blueBtn!!.addActionListener(actionListener)
        greenBtn = JButton("Green")
        greenBtn!!.addActionListener(actionListener)
        redBtn = JButton("Red")
        redBtn!!.addActionListener(actionListener)
        magentaBtn = JButton("Magenta")
        magentaBtn!!.addActionListener(actionListener)

        // add to panel
        controls.add(greenBtn)
        controls.add(blueBtn)
        controls.add(blackBtn)
        controls.add(redBtn)
        controls.add(magentaBtn)
        controls.add(clearBtn)

        // add to content pane
        content.add(controls, BorderLayout.NORTH)
        frame.setSize(600, 600)
        // can close frame
        frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
        // show the swing paint result
        frame.isVisible = true

        // Now you can try our Swing Paint !!! Enjoy :D
    }

}