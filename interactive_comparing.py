#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
ZetCode PyQt5 tutorial

This example shows an icon
in the titlebar of the window.

Author: Jan Bodnar
Website: zetcode.com
Last edited: August 2017
"""
import json
import sys
import numpy as np
import os

from PIL import Image
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QWidget, QPushButton,
    QHBoxLayout, QVBoxLayout, QApplication, QSlider, QLabel)
from PyQt5.QtCore import Qt
from scipy.ndimage import gaussian_filter


class Example(QWidget):
    image_size = (200, 200)

    def __init__(self, images_path):
        super().__init__()

        self.im_path = images_path

        self.prepare_data()
        self.initUI()

        self.index = 0
        self.update()

    def initUI(self):
        self.setGeometry(300, 300, 300, 220)
        self.setWindowTitle('Explanation compare dialog')

        # define navigation controls and labels
        prew_button = QPushButton("<<")
        prew_button.clicked.connect(self.prew_clicked)
        next_button = QPushButton(">>")
        next_button.clicked.connect(self.next_clicked)
        self.sl = QSlider(Qt.Horizontal)
        self.sl.valueChanged.connect(self.replot_masked_images)
        self.threshold_label = QLabel()
        self.threshold_label.setText(str(0))
        self.correct_class_label = QLabel()
        self.predicted_class_label = QLabel()
        predicted_label = QLabel("Predicted:")
        true_label = QLabel("True:")
        self.image_name_label = QLabel()

        # images
        self.pics = [QLabel(self) for _ in self.approaches]
        self.qt_masks = [QLabel(self) for _ in self.approaches]
        self.qt_masks_ex = [QLabel(self) for _ in self.approaches]

        # place images in grid
        im_verticals = []
        for i, pic in enumerate(self.pics):
            im_vertical = QVBoxLayout()
            im_vertical.addWidget(self.qt_masks[i])
            im_vertical.addWidget(self.qt_masks_ex[i])
            im_vertical.addWidget(pic)

            pic.setAlignment(Qt.AlignCenter)
            self.qt_masks[i].setAlignment(Qt.AlignCenter)
            self.qt_masks_ex[i].setAlignment(Qt.AlignCenter)

            app_label = QLabel()
            app_label.setText(self.approaches[i])
            app_label.setAlignment(Qt.AlignCenter)
            im_vertical.addWidget(app_label)
            im_verticals.append(im_vertical)

        # place labels in widget
        hbox_txt = QHBoxLayout()
        hbox_txt.addWidget(self.image_name_label)
        hbox_txt.addStretch(5)
        hbox_txt.addWidget(true_label)
        hbox_txt.addWidget(self.correct_class_label)
        hbox_txt.addStretch(1)
        hbox_txt.addWidget(predicted_label)
        hbox_txt.addWidget(self.predicted_class_label)
        hbox_txt.setAlignment(Qt.AlignCenter)

        # place images box in widget
        hbox_im = QHBoxLayout()
        hbox_im.setAlignment(Qt.AlignCenter)
        for apps in im_verticals:
            hbox_im.addLayout(apps)

        # hbox for buttons
        hbox = QHBoxLayout()
        hbox.addWidget(prew_button)
        hbox.addWidget(self.sl)
        hbox.addWidget(self.threshold_label)
        hbox.addWidget(next_button)

        # color controls
        self.r_slider = QSlider(Qt.Horizontal)
        self.g_slider = QSlider(Qt.Horizontal)
        self.b_slider = QSlider(Qt.Horizontal)

        self.r_slider.valueChanged.connect(self.replot_masked_images)
        self.g_slider.valueChanged.connect(self.replot_masked_images)
        self.b_slider.valueChanged.connect(self.replot_masked_images)

        self.sigma_slider = QSlider(Qt.Horizontal)
        self.sigma_slider.valueChanged.connect(self.update)

        hbox_color = QHBoxLayout()
        hbox_color.addWidget(QLabel("Red"))
        hbox_color.addWidget(self.r_slider)
        hbox_color.addWidget(QLabel("Green"))
        hbox_color.addWidget(self.g_slider)
        hbox_color.addWidget(QLabel("Blue"))
        hbox_color.addWidget(self.b_slider)
        hbox_color.addWidget(QLabel("Sigma"))
        hbox_color.addWidget(self.sigma_slider)

        # main widget
        vbox = QVBoxLayout()
        vbox.addLayout(hbox_txt)
        vbox.addLayout(hbox_im)
        vbox.addLayout(hbox)
        vbox.addLayout(hbox_color)

        self.setLayout(vbox)
        self.show()

    def prepare_data(self):
        # open class labels
        try:
            with open(os.path.join(self.im_path, "classes.json"), "r") as f:
                classes = json.load(f)
                classes = {int(v): k for k, v in classes.items()}
        except FileNotFoundError:
            classes = None

        # list approaches
        self.approaches = sorted(
            list(set(os.listdir(self.im_path)) -
                 {"all", "original", ".DS_Store", "classes.json"}))
        # list images
        self.images = os.listdir(os.path.join(self.im_path, "all"))
        self.im_names = ["-".join(x.split("-")[:-2]) for x in self.images]

        try:
            self.predicted_values = [classes[int(x.split(
                "-")[-2])] for x in self.images]
            self.true_values = [classes[int(
                x.split("-")[-1].split(".")[0])] for x in self.images]
        except ValueError:
            self.predicted_values = [x.split(
                "-")[-2] for x in self.images]
            self.true_values = [
                x.split("-")[-1].split(".")[0] for x in self.images]


    def update(self):
        # change image name label
        self.image_name_label.setText(self.images[self.index])

        # load masks and pre-process them
        self.masks = [np.array(Image.open(
            os.path.join(
                self.im_path, a, "pos", self.images[self.index])).resize(
            self.image_size)) / 255.0 for a in self.approaches]
        sigma = self.sigma_slider.value() / 100 * 10
        self.masks_extended = [gaussian_filter(
            x.copy(), sigma=sigma) for x in self.masks]

        # open image
        self.image = Image.open(
            os.path.join(self.im_path, "original", self.im_names[self.index]))

        self.image = self.image.resize(self.image_size)
        self.image = np.array(self.image)

        # plot masks
        for i, (mask, mask_ex) in enumerate(
                zip(self.masks, self.masks_extended)):
            mask = (np.stack((mask,)*3, -1) * 255).astype(np.uint8)
            qimage = QImage(mask, mask.shape[1], mask.shape[0],
                        QImage.Format_RGB888)
            pixmap = QPixmap(qimage)
            self.qt_masks[i].setPixmap(pixmap)

            mask_ex = (np.stack((mask_ex,)*3, -1) * 255).astype(np.uint8)
            qimage = QImage(mask_ex, mask_ex.shape[1], mask_ex.shape[0],
                        QImage.Format_RGB888)
            pixmap = QPixmap(qimage)
            self.qt_masks_ex[i].setPixmap(pixmap)

        # plot masked images
        self.replot_masked_images()

        # classes
        self.predicted_class_label.setText(self.predicted_values[self.index])
        self.correct_class_label.setText(self.true_values[self.index])

    def prew_clicked(self):
        self.index -= 1
        self.index = self.index % len(self.images)
        self.update()

    def next_clicked(self):
        self.index += 1
        self.index = self.index % len(self.images)
        self.update()

    def replot_masked_images(self):
        threshold = self.sl.value() / 100
        self.threshold_label.setText(str(threshold))

        for i, mask in enumerate(self.masks_extended):
            im = self.image.copy()
            im[mask < threshold, 0] = int(self.r_slider.value() / 100 * 255)
            im[mask < threshold, 1] = int(self.g_slider.value() / 100 * 255)
            im[mask < threshold, 2] = int(self.b_slider.value() / 100 * 255)

            qimage = QImage(im, im.shape[1],
                            im.shape[0],
                            QImage.Format_RGB888)
            pixmap = QPixmap(qimage)
            self.pics[i].setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example(sys.argv[1])
    sys.exit(app.exec_())
