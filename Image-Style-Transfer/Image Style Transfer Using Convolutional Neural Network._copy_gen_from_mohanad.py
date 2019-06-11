"""Image Style Transfer Using Convolutional Neural Network
code Written in python, Ui made with PyQt5"""
import skimage
import skimage.io
import skimage.transform
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
import logo
import threading
from keras.preprocessing import image
import tensorflow as tf

from keras.preprocessing.image import load_img

# global variables created to control the UI and code parameters.
global content_path
global style_path
global outputImage
global pixmap
global exitflag
exitflag=0
global flag1
flag1=0
global flag2
flag2=0
global flag3
flag3=0
global count
count=0
global iter
iter = 0
DECAY = .9
EPSILON = 1e-8

"""Ui_MainWindow is the main class of the UI,
all UI parameters and code functions defined here."""
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(946,600)
        MainWindow.setStyleSheet("font: 75 22pt \"MS Shell Dlg 2\";")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 946, 800))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(":/logo/grey-background-v1.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.mainlog = QtWidgets.QLabel(self.centralwidget)
        self.mainlog.setGeometry(QtCore.QRect(290, 80, 411, 251))
        self.mainlog.setText("")
        self.mainlog.setPixmap(QtGui.QPixmap(":/logo/logo.png"))
        self.mainlog.setScaledContents(True)
        self.mainlog.setObjectName("mainlog")
        self.About = QtWidgets.QLabel(self.centralwidget)
        self.About.setGeometry(QtCore.QRect(20, 0, 711, 501))
        self.About.setText("")
        self.About.setPixmap(QtGui.QPixmap(":/logo/AOUT.png"))
        self.About.setScaledContents(True)
        self.About.setObjectName("About")
        self.About.hide()
        self.smalllogo = QtWidgets.QLabel(self.centralwidget)
        self.smalllogo.setGeometry(QtCore.QRect(440, 480, 91, 51))
        self.smalllogo.setText("")
        self.smalllogo.setPixmap(QtGui.QPixmap(":/logo/logo.png"))
        self.smalllogo.setScaledContents(True)
        self.smalllogo.setObjectName("smalllogo")
        self.smalllogo.hide()
        self.contentbutton = QtWidgets.QPushButton(self.centralwidget)
        self.contentbutton.setGeometry(QtCore.QRect(60, 40, 151, 41))
        self.contentbutton.setStyleSheet("font: 75 12pt \"News706 BT\";")
        self.contentbutton.setObjectName("contentbutton")
        self.contentbutton.hide()
        self.stylebutton = QtWidgets.QPushButton(self.centralwidget)
        self.stylebutton.setGeometry(QtCore.QRect(400, 40, 151, 41))
        self.stylebutton.setStyleSheet("font: 75 12pt \"News706 BT\";")
        self.stylebutton.setObjectName("stylebutton")
        self.stylebutton.hide()
        self.generatebutton = QtWidgets.QPushButton(self.centralwidget)
        self.generatebutton.setGeometry(QtCore.QRect(400, 400, 151, 41))
        self.generatebutton.setStyleSheet("font: 75 12pt \"News706 BT\";")
        self.generatebutton.setObjectName("generatebutton")
        self.generatebutton.hide()
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(280, 400, 491, 31))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.hide()
        self.contentframe = QtWidgets.QLabel(self.centralwidget)
        self.contentframe.setGeometry(QtCore.QRect(10, 90, 251, 191))
        self.contentframe.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.contentframe.setText("")
        self.contentframe.setPixmap(QtGui.QPixmap(":/logo/image.png"))
        self.contentframe.setScaledContents(True)
        self.contentframe.setObjectName("contentframe")
        self.contentframe.hide()
        self.styleframe = QtWidgets.QLabel(self.centralwidget)
        self.styleframe.setGeometry(QtCore.QRect(350, 90, 251, 191))
        self.styleframe.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.styleframe.setText("")
        self.styleframe.setPixmap(QtGui.QPixmap(":/logo/image.png"))
        self.styleframe.setScaledContents(True)
        self.styleframe.setObjectName("styleframe")
        self.styleframe.hide()
        self.outputframe = QtWidgets.QLabel(self.centralwidget)
        self.outputframe.setGeometry(QtCore.QRect(680, 90, 251, 191))
        self.outputframe.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.outputframe.setText("")
        self.outputframe.setPixmap(QtGui.QPixmap(":/logo/qustion.png"))
        self.outputframe.setScaledContents(True)
        self.outputframe.setObjectName("outputframe")
        self.outputframe.hide()
        self.savebutton = QtWidgets.QPushButton(self.centralwidget)
        self.savebutton.setGeometry(QtCore.QRect(730, 40, 151, 41))
        self.savebutton.setStyleSheet("font: 75 12pt \"News706 BT\";")
        self.savebutton.setObjectName("savebutton")
        self.savebutton.hide()
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(280, 330, 121, 31))
        self.comboBox.setStyleSheet("font: 75 14pt \"MS Shell Dlg 2\";")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.hide()
        self.qualty = QtWidgets.QLabel(self.centralwidget)
        self.qualty.setGeometry(QtCore.QRect(170, 320, 111, 41))
        self.qualty.setObjectName("qualty")
        self.qualty.hide()
        self.when = QtWidgets.QLabel(self.centralwidget)
        self.when.setGeometry(QtCore.QRect(195, 340, 571, 121))
        self.when.setText("")
        self.when.setPixmap(QtGui.QPixmap(":/logo/when.png"))
        self.when.setScaledContents(True)
        self.when.setObjectName("when")
        self.resbox = QtWidgets.QComboBox(self.centralwidget)
        self.resbox.setGeometry(QtCore.QRect(660, 330, 121, 31))
        self.resbox.setStyleSheet("font: 75 14pt \"MS Shell Dlg 2\";")
        self.resbox.setObjectName("resbox")
        self.resbox.addItem("")
        self.resbox.addItem("")
        self.resbox.addItem("")
        self.resbox.addItem("")
        self.resbox.hide()
        self.res = QtWidgets.QLabel(self.centralwidget)
        self.res.setGeometry(QtCore.QRect(510, 320, 141, 41))
        self.res.setObjectName("res")
        self.res.hide()
        self.warninglabel = QtWidgets.QLabel(self.centralwidget)
        self.warninglabel.setGeometry(QtCore.QRect(260, 390, 421, 61))
        self.warninglabel.setText("")
        self.warninglabel.setPixmap(QtGui.QPixmap(":/logo/warning.png"))
        self.warninglabel.setScaledContents(True)
        self.warninglabel.setObjectName("warninglabel")
        self.warninglabel.hide()
        self.equalabel = QtWidgets.QLabel(self.centralwidget)
        self.equalabel.setGeometry(QtCore.QRect(610, 150, 61, 71))
        self.equalabel.setText("")
        self.equalabel.setPixmap(QtGui.QPixmap(":/logo/equal.png"))
        self.equalabel.setScaledContents(True)
        self.equalabel.setObjectName("equalabel")
        self.equalabel.hide()
        self.pluslabel = QtWidgets.QLabel(self.centralwidget)
        self.pluslabel.setGeometry(QtCore.QRect(260, 140, 91, 81))
        self.pluslabel.setText("")
        self.pluslabel.setPixmap(QtGui.QPixmap(":/logo/plus-big-512.png"))
        self.pluslabel.setScaledContents(True)
        self.pluslabel.setObjectName("pluslabel")
        self.pluslabel.hide()
        self.mainlog.raise_()
        self.About.raise_()
        self.when.raise_()
        self.progressBar.raise_()
        self.smalllogo.raise_()
        self.contentbutton.raise_()
        self.stylebutton.raise_()
        self.generatebutton.raise_()
        self.contentframe.raise_()
        self.styleframe.raise_()
        self.outputframe.raise_()
        self.savebutton.raise_()
        self.comboBox.raise_()
        self.qualty.raise_()
        self.resbox.raise_()
        self.res.raise_()
        self.warninglabel.raise_()
        self.equalabel.raise_()
        self.pluslabel.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 946, 41))
        self.menubar.setObjectName("menubar")
        self.menuHome = QtWidgets.QMenu(self.menubar)
        self.menuHome.setObjectName("menuHome")
        self.menuCreate_New = QtWidgets.QMenu(self.menubar)
        self.menuCreate_New.setObjectName("menuCreate_New")
        self.menuAbout = QtWidgets.QMenu(self.menubar)
        self.menuAbout.setObjectName("menuAbout")
        self.menuExit = QtWidgets.QMenu(self.menubar)
        self.menuExit.setObjectName("menuExit")
        MainWindow.setMenuBar(self.menubar)
        self.actionCreate_New = QtWidgets.QAction(MainWindow)
        self.actionCreate_New.setObjectName("actionCreate_New")
        self.actionHome = QtWidgets.QAction(MainWindow)
        self.actionHome.setObjectName("actionHome")
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionExit2 = QtWidgets.QAction(MainWindow)
        self.actionExit2.setObjectName("actionExit2")
        self.menuHome.addAction(self.actionHome)
        self.menuHome.addAction(self.actionExit2)
        self.menuCreate_New.addSeparator()
        self.menuCreate_New.addAction(self.actionCreate_New)
        self.menuAbout.addAction(self.actionAbout)
        self.menuExit.addAction(self.actionExit)
        self.menubar.addAction(self.menuHome.menuAction())
        self.menubar.addAction(self.menuCreate_New.menuAction())
        self.menubar.addAction(self.menuAbout.menuAction())
        self.menubar.addAction(self.menuExit.menuAction())
        self.retranslateUi(MainWindow)
        self.actionCreate_New.triggered.connect(self.mainlog.hide)
        self.actionHome.triggered.connect(self.progressBar.hide)
        self.actionHome.triggered.connect(self.contentbutton.hide)
        self.actionHome.triggered.connect(self.generatebutton.hide)
        self.actionHome.triggered.connect(self.stylebutton.hide)
        self.actionHome.triggered.connect(self.smalllogo.hide)
        self.actionHome.triggered.connect(self.About.hide)
        self.actionHome.triggered.connect(self.qualty.hide)
        self.actionHome.triggered.connect(self.savebutton.hide)
        self.actionHome.triggered.connect(self.comboBox.hide)
        self.actionHome.triggered.connect(self.warninglabel.hide)
        self.actionHome.triggered.connect(self.pluslabel.hide)
        self.actionHome.triggered.connect(self.equalabel.hide)
        self.actionHome.triggered.connect(self.outputframe.hide)
        self.actionHome.triggered.connect(self.res.hide)
        self.actionHome.triggered.connect(self.resbox.hide)
        self.actionHome.triggered.connect(self.mainlog.show)
        self.actionHome.triggered.connect(self.when.show)
        self.actionCreate_New.triggered.connect(self.createNewScreen)
        self.actionAbout.triggered.connect(self.About.show)
        self.actionAbout.triggered.connect(self.smalllogo.show)
        self.actionAbout.triggered.connect(self.generatebutton.hide)
        self.actionAbout.triggered.connect(self.warninglabel.hide)
        self.actionAbout.triggered.connect(self.progressBar.hide)
        self.generatebutton.clicked.connect(self.generatebutton.hide)
        self.actionAbout.triggered.connect(self.contentbutton.hide)
        self.actionAbout.triggered.connect(self.stylebutton.hide)
        self.actionAbout.triggered.connect(self.pluslabel.hide)
        self.actionAbout.triggered.connect(self.equalabel.hide)
        self.actionAbout.triggered.connect(self.outputframe.hide)
        self.actionAbout.triggered.connect(self.when.hide)
        self.actionAbout.triggered.connect(self.qualty.hide)
        self.actionAbout.triggered.connect(self.savebutton.hide)
        self.actionAbout.triggered.connect(self.comboBox.hide)
        self.actionAbout.triggered.connect(self.res.hide)
        self.actionAbout.triggered.connect(self.resbox.hide)
        self.actionAbout.triggered.connect(self.mainlog.hide)
        self.actionExit2.triggered.connect(self.exit)
        self.actionExit.triggered.connect(self.openhelp)
        self.actionAbout.triggered.connect(self.contentframe.hide)
        self.actionAbout.triggered.connect(self.styleframe.hide)
        self.actionHome.triggered.connect(self.contentframe.hide)
        self.actionHome.triggered.connect(self.styleframe.hide)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.contentbutton.clicked.connect(self.setContentImage)
        self.stylebutton.clicked.connect(self.setStyleImage)
        self.generatebutton.clicked.connect(self.lunch_thread)
        self.savebutton.clicked.connect(self.saveimage)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("artme", "artme"))
        self.contentbutton.setText(_translate("MainWindow", "Content Image"))
        self.stylebutton.setText(_translate("MainWindow", "Style Image"))
        self.generatebutton.setText(_translate("MainWindow", "Generate"))
        self.savebutton.setText(_translate("MainWindow", "Save Image"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Low"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Medium"))
        self.comboBox.setItemText(2, _translate("MainWindow", "High"))
        self.qualty.setText(_translate("MainWindow", "Quality: "))
        self.resbox.setItemText(0, _translate("MainWindow", "256 Px"))
        self.resbox.setItemText(1, _translate("MainWindow", "512 Px"))
        self.resbox.setItemText(2, _translate("MainWindow", "1024 Px"))
        self.resbox.setItemText(3, _translate("MainWindow", "2048 Px"))
        self.res.setText(_translate("MainWindow", "Resolution:"))
        self.menuHome.setTitle(_translate("MainWindow", "Home"))
        self.menuCreate_New.setTitle(_translate("MainWindow", "Create New"))
        self.menuAbout.setTitle(_translate("MainWindow", "About"))
        self.menuExit.setTitle(_translate("MainWindow", "Help"))
        self.actionCreate_New.setText(_translate("MainWindow", "Create New"))
        self.actionHome.setText(_translate("MainWindow", "Home"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.actionExit.setText(_translate("MainWindow", "Help"))
        self.actionExit2.setText(_translate("MainWindow", "Exit"))

    # openhelp function open the help file.
    def openhelp(self):
        import os
        filename = 'Help.pdf'
        try:
            os.startfile(filename)
        except:
            return

    # createNewScreen function control of what shows in the create new screen.
    def createNewScreen(self):
        global flag3
        if(flag3==1):
            self.savebutton.show()
        else:
            self.savebutton.hide()
        self.smalllogo.show()
        self.contentframe.show()
        self.styleframe.show()
        self.contentbutton.show()
        self.warninglabel.hide()
        self.generatebutton.show()
        self.qualty.show()
        self.comboBox.show()
        self.res.show()
        self.resbox.show()
        self.pluslabel.show()
        self.equalabel.show()
        self.outputframe.show()
        self.mainlog.hide()
        self.when.hide()
        self.About.hide()
        self.progressBar.hide()
        self.stylebutton.show()
        self.contentframe.show()
        self.styleframe.show()

    """onCountChanged function control on updating the progrssBar."""
    def onCountChanged(self, value):
        self.progressBar.setValue(value)

    """setContentImage function control on choosing the content image."""
    def setContentImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileNames(None, "Select Image", "",
                                                             "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fileName:
            global content_path
            content_path = fileName[0]
            pixmap = QtGui.QPixmap(fileName[0])
            pixmap = pixmap.scaled(290, 290, QtCore.Qt.KeepAspectRatio)
            self.contentframe.setPixmap(pixmap)
            self.contentframe.setAlignment(QtCore.Qt.AlignCenter)
            global flag1
            flag1 =1
            global flag2
            if (flag1==1 and flag2==1):
                self.outputframe.show()
                self.warninglabel.hide()
                self.generatebutton.show()
                self.pluslabel.show()
                self.equalabel.show()

    """setStyleImage function control on choosing the style image."""
    def setStyleImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileNames(None, "Select Image", "",
                                                             "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fileName:
            global style_path
            style_path = fileName[0]
            pixmap = QtGui.QPixmap(fileName[0])
            pixmap = pixmap.scaled(290, 290, QtCore.Qt.KeepAspectRatio)
            self.styleframe.setPixmap(pixmap)
            self.styleframe.setAlignment(QtCore.Qt.AlignCenter)
            global flag2
            flag2 = 1
            global flag1
            if (flag2==1 and flag1==1):
                self.outputframe.show()
                self.warninglabel.hide()
                self.generatebutton.show()
                self.pluslabel.show()
                self.equalabel.show()

    """Generate function is start when the Generate button pushed. it start the main algorithm."""
    def Generate(self):
        global outputImage
        global exitflag
        exitflag=1
        global flag1
        global flag2
        if (flag1 == 0 or flag2 == 0):
            self.warninglabel.show()
            return
        self.actionHome.setEnabled(False)
        self.actionCreate_New.setEnabled(False)
        self.actionAbout.setEnabled(False)
        self.outputframe.setPixmap(QtGui.QPixmap(":/logo/qustion.png"))
        self.savebutton.hide()
        self.progressBar.setValue(0)
        self.progressBar.show()
        # iter control the number of iteration the algorithm run, the user choose it.
        global  iter
        iter=0
        if self.comboBox.currentText() == 'Low':
            iter=100
        elif self.comboBox.currentText() == 'Medium':
            iter=500
        else:
            iter=1000

        # resulotion control the output image resulotion, the user choose it.
        resolution = 0
        if self.resbox.currentText() == '256 Px':
            resolution = 256
        elif self.resbox.currentText() == '512 Px':
            resolution = 512
        elif self.resbox.currentText() == '1024 Px':
            resolution = 1024
        else:
            resolution = 2048
        # outputImage get the result from the MainFunc.
        outputImage = self.MainFunc(content_path, style_path, iter, resolution)# here we start to generate
        pixmap = QtGui.QPixmap(outputImage.toqpixmap())
        pixmap = pixmap.scaled(290, 290, QtCore.Qt.KeepAspectRatio)
        self.outputframe.setPixmap(pixmap)
        self.outputframe.setAlignment(QtCore.Qt.AlignCenter)
        self.outputframe.show()
        self.savebutton.show()
        global flag3
        flag3 = 1
        self.actionHome.setEnabled(True)
        self.actionCreate_New.setEnabled(True)
        self.actionAbout.setEnabled(True)

    """lunch_thread control the start of the second thread that running the MainFunc."""
    def lunch_thread(self):
        t = threading.Thread(target=self.Generate)
        t.start()

    """saveimage function control the saving of the output image."""
    def saveimage(self):
        global outputImage
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Select Image", "",
                                                             "Image Files (*.jpg *.png *.jpeg *.bmp)")
        if(fileName):
            outputImage.save(fileName)

    """exit function control on exit the application."""
    def exit(self):
        if(exitflag == 1):
            self.exit()
        else:
            exit(1)

    """MainFunc is the main function that running the main algorithm"""
    def MainFunc(self, content_path, style_path, iter, resolution):
        import numpy as np
        from PIL import Image
        import tensorflow as tf
        import tensorflow.contrib.eager as tfe
        from tensorflow.python.keras.preprocessing import image as kp_image
        from tensorflow.python.keras import models

        # Eager execution is a flexible machine learning platform for research and experimentation.
        # Since we're using eager our model is callable just like any other function.
        tf.enable_eager_execution()
        print("Eager execution: {}".format(tf.executing_eagerly()))

        # define calc to the external thread.
        self.calc = External()
        self.calc.countChanged.connect(self.progressBar.setValue)

        # Content layer for the feature maps
        content_layers = ['block5_conv2']

        # Style layer for the feature maps.
        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1'
                        ]

        num_content_layers = len(content_layers)
        num_style_layers = len(style_layers)



        # load_img function get the path of the image,
        # resize it and broadcast the image array such that it has a batch dimension.
        def load_img(path_to_img):
            max_dim = resolution
            img = Image.open(path_to_img)
            long = max(img.size)
            scale = max_dim / long
            img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)
            img = kp_image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            return img

        def get_resnet_model(img_path_):
            from keras.preprocessing.image import img_to_array
            from keras.preprocessing import image
            from keras_applications import imagenet_utils
            from keras.preprocessing import image

            model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
            # load the image
            img_path = img_path_
            img = image.load_img(img_path, target_size=(224, 224))
            x =img_to_array(img)
            # ensure we have a 4d tensor with single element in the batch dimension,
            # the preprocess the input for prediction using resnet50

            x = np.expand_dims(x, axis=0)
            x = imagenet_utils.preprocess_input(x,'channels_first')

            x = model(x)
            return x



        # load_and_process_img is charge on load the image into the vgg16 network.
        def load_and_process_img(path_to_img ,flag = False, image = None):
            img = load_img(path_to_img)

            if flag == True:
                with tf.Session() as sess:
                    if image == None:
                        img, _ = load_img_to(path_to_img)
                        img = tf.convert_to_tensor(img, dtype=tf.float32)
                        img = tf.expand_dims(img, axis=0)
                    else:
                        img = image
                    # Initialize new generative net
                    with tf.variable_scope('generator'):
                        #gen = Generator()
                        #img = gen.build(tf.convert_to_tensor(img))
                        img = get_resnet_model(path_to_img)

                        #sess.run(tf.global_variables_initializer())
                    #saver = tf.train.Saver()
                    #saver.restore(sess, )
                    #img = sess.run(img)

            img = tf.keras.applications.vgg16.preprocess_input(img)
            return img

        def deprocess_img(processed_img):
            x = processed_img.copy()
            if len(x.shape) == 4:
                x = np.squeeze(x, 0)
            assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                                       "dimension [1, height, width, channel] or [height, width, channel]")
            if len(x.shape) != 3:
                raise ValueError("Invalid input to deprocessing image")

            x[:, :, 0] += 103.939
            x[:, :, 1] += 116.779
            x[:, :, 2] += 123.68
            x = x[:, :, ::-1]

            x = np.clip(x, 0, 255).astype('uint8')
            return x



        # get_model function load the VGG16 model and access the intermediate layers.
        # Returns: a Keras model that takes image inputs and outputs the style and content intermediate layers.
        def get_model():
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            # We load pretrained VGG Network, trained on imagenet data
            vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
            vgg.trainable = False
            # Get output layers corresponding to style and content layers
            style_outputs = [vgg.get_layer(name).output for name in style_layers]
            content_outputs = [vgg.get_layer(name).output for name in content_layers]
            model_outputs = style_outputs + content_outputs
            # Build model
            return models.Model(vgg.input, model_outputs)

        # get_content_loss function calculate the content loss that is the
        # Mean Squared Error between the two feature representations matrices.
        def get_content_loss(base_content, target):
            return tf.reduce_mean(tf.square(base_content - target))

        # Calculate the gram matrix for the style representation.
        def gram_matrix(input_tensor):
            # Make the image channels
            channels = int(input_tensor.shape[-1])
            a = tf.reshape(input_tensor, [-1, channels])
            n = tf.shape(a)[0]
            gram = tf.matmul(a, a, transpose_a=True)
            return gram / tf.cast(n, tf.float32)

        # get the style loss by calculate the Mean Squared Error between the two gram matrices.
        # We scale the loss at a given layer by the size of the feature map and the number of filters
        def get_style_loss(base_style, gram_target):
            height, width, channels = base_style.get_shape().as_list()
            gram_style = gram_matrix(base_style)
            return tf.reduce_mean(tf.square(gram_style - gram_target))

        """This function will simply load and preprocess both the content and style
            images from their path. Then it will feed them through the network to obtain
            the outputs of the intermediate layers.
            Returns the style and the content features representation."""
        def get_feature_representations(model, content_path, style_path):
            # Load our images into the VGG16 Network
            content_image = load_and_process_img(content_path)
            style_image = load_and_process_img(style_path)

            # compute content and style features
            style_outputs = model(style_image)
            content_outputs = model(content_image)

            # Get the style and content feature representations from our model
            style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
            content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
            return style_features, content_features

        """This function compute the content, style and total loss.
            we use model that will give us access to the intermediate layers."""
        def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
            style_weight, content_weight = loss_weights

            # Feed our init image through our model. This will give us the content and
            # style representations at our desired layers.
            model_outputs = model(init_image)

            style_output_features = model_outputs[:num_style_layers]
            content_output_features = model_outputs[num_style_layers:]

            style_score = 0
            content_score = 0

            # calculate the style losses from all layers
            # equally weight each contribution of each loss layer
            weight_per_style_layer = 1.0 / float(num_style_layers)
            for target_style, comb_style in zip(gram_style_features, style_output_features):
                style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

            # calculate content losses from all layers
            weight_per_content_layer = 1.0 / float(num_content_layers)
            for target_content, comb_content in zip(content_features, content_output_features):
                content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

            style_score *= style_weight
            content_score *= content_weight

            # Get total loss
            loss = style_score + content_score
            return loss, style_score, content_score

        # Compute gradients according to input image
        def compute_grads(cfg):
            with tf.GradientTape() as tape:
                all_loss = compute_loss(**cfg)
            total_loss = all_loss[0]
            return tape.gradient(total_loss, cfg['init_image']), all_loss

        """The main method of the code, running the main loop for generating the image."""
        def run_style_transfer(content_path,
                               style_path,
                               num_iterations=1000,
                               content_weight=1e3,
                               style_weight=1e-2):
            # We don't train any layers of our model, so we set their trainable to false.
            model = get_model()
            for layer in model.layers:
                layer.trainable = False

            # Get the style and content feature representations (from our specified intermediate layers)
            style_features, content_features = get_feature_representations(model, content_path, style_path)
            gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

            # Set initial image
            init_image = load_and_process_img(content_path, flag=True)
            init_image = tfe.Variable(init_image, dtype=tf.float32)
            # We  use Adam Optimizer
            opt = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.99, epsilon=1e-1)

            # Store our best result
            best_loss, best_img = float('inf'), None

            # Create config
            loss_weights = (style_weight, content_weight)
            cfg = {
                'model': model,
                'loss_weights': loss_weights,
                'init_image': init_image,
                'gram_style_features': gram_style_features,
                'content_features': content_features
            }

            norm_means = np.array([103.939, 116.779, 123.68])
            min_vals = -norm_means
            max_vals = 255 - norm_means

            # Main loop
            for i in range(num_iterations):
                global count
                count=i
                self.calc.start()
                print(i)

                grads, all_loss = compute_grads(cfg)
                loss, style_score, content_score = all_loss
                opt.apply_gradients([(grads, init_image)])
                clipped = tf.clip_by_value(init_image, min_vals, max_vals)
                init_image.assign(clipped)

                if loss < best_loss:
                    # Update best loss and best image from total loss.
                    best_loss = loss
                    best_img = deprocess_img(init_image.numpy())

            return best_img, best_loss

        best, best_loss = run_style_transfer(content_path, style_path, num_iterations=iter)
        im = Image.fromarray(best)
        return im

"""External class control the thread running the ProgressBar."""
class External(QThread):
    countChanged = pyqtSignal(int)

    def run(self):
        global count
        global iter
        ii =((count + 1) / iter) * 100
        self.countChanged.emit(ii)


###################
#generator
###################

import tensorflow as tf

# Model hyper-parameters
DECAY = .9
EPSILON = 1e-8


class Generator:
    """Generative model with the architectural specifications suited for artistic style transfer."""

    def __init__(self, is_training=True):
        self.training = is_training

    def build(self, img):
        """Constructs the generative network's layers. Normally called after initialization.

        Args:
            img: 4D tensor representation of image batch
        """

        self.padded = self._pad(img, 40)

        self.conv1 = self._conv_block(self.padded, maps_shape=[9, 9, 3, 32], stride=1, name='conv1')
        self.conv2 = self._conv_block(self.conv1, maps_shape=[3, 3, 32, 64], stride=2, name='conv2')
        self.conv3 = self._conv_block(self.conv2, maps_shape=[3, 3, 64, 128], stride=2, name='conv3')

        self.resid1 = self._residual_block(self.conv3, maps_shape=[3, 3, 128, 128], stride=1, name='resid1')
        self.resid2 = self._residual_block(self.resid1, maps_shape=[3, 3, 128, 128], stride=1, name='resid2')
        self.resid3 = self._residual_block(self.resid2, maps_shape=[3, 3, 128, 128], stride=1, name='resid3')
        self.resid4 = self._residual_block(self.resid3, maps_shape=[3, 3, 128, 128], stride=1, name='resid4')
        self.resid5 = self._residual_block(self.resid4, maps_shape=[3, 3, 128, 128], stride=1, name='resid5')

        self.conv4 = self._upsample_block(self.resid5, maps_shape=[3, 3, 64, 128], stride=2, name='conv4')
        self.conv5 = self._upsample_block(self.conv4, maps_shape=[3, 3, 32, 64], stride=2, name='conv5')
        self.conv6 = self._conv_block(self.conv5, maps_shape=[9, 9, 32, 3], stride=1, name='conv6', activation=None)

        self.output = tf.nn.sigmoid(self.conv6) #TO-DO: understand why the tanh function not working !!
        # self.output = tf.nn.tanh(self.conv6)
        return self.output

    @staticmethod
    def _get_weights(shape):
        """Returns a variable for weights with a specified filters shape.

        Args:
            shape: a list specifying the initialized weights shape

        Returns:
            weights: tf.Variable representing a set of weights with a normal distribution
        """
        #truncated_normal= Outputs random values from a truncated normal distribution.
        init = tf.truncated_normal(shape, mean=0., stddev=.1)
        weights = tf.Variable(init, dtype=tf.float32)
        return weights

    @staticmethod
    def _instance_normalize(inputs):
        """Instance normalize inputs to reduce covariate shift and reduce dependency on input contrast to improve results.

        Args:
            inputs: 4D tensor representing image layer encodings

        Returns:
            maps: 4D tensor of batch normalized inputs
        """

        with tf.variable_scope('instance_normalization'):
            batch, height, width, channels = [_.value for _ in inputs.get_shape()]
            mu, sigma_sq = tf.nn.moments(inputs, [1, 2], keep_dims=True)

            shift = tf.Variable(tf.constant(.1, shape=[channels]))
            scale = tf.Variable(tf.ones([channels]))
            normalized = (inputs - mu) / (sigma_sq + EPSILON) ** .5
            maps = scale * normalized + shift
            return maps

    @staticmethod
    def _pad(inputs, size):
        """Pads input of the image so the output is the same dimensions even after strided convolution.

        Args:
            inputs: 4D tensor representing image layer encodings
            size: int specifying the pad size

        Returns:
            padded_inputs: 4D tensor of padded inputs
        """

        padded_inputs = tf.pad(inputs, [[0, 0], [size, size], [size, size], [0, 0]], "REFLECT")
        return padded_inputs

    @staticmethod
    def _batch_normalize(inputs, num_maps, is_training):
        """Batch normalize inputs to reduce covariate shift and improve the efficiency of training.

        Args:
            inputs: 4D tensor representing image layer encodings
            num_maps: int representing the number of input feature maps
            is_training: bool representing whether or not the model is
                         being trained rather than being used for inference

        Returns:
            bn_inputs: 4D tensor of batch normalized inputs
        """

        with tf.variable_scope("batch_normalization"):
            #   variables for scaling and offsetting our inputs
            scale = tf.Variable(tf.ones([num_maps], dtype=tf.float32))
            offset = tf.Variable(tf.zeros([num_maps], dtype=tf.float32))

            # Mean and variances related to our current batch
            #mean and varience of the input (image) matrix
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])

            # Create an optimizer to maintain a 'moving average'
            #by maintains moving averages of variables by employing an exponential decay.
            #Evaluations that use averaged parameters sometimes produce
            # significantly better results than the final trained values.
            ema = tf.train.ExponentialMovingAverage(decay=DECAY)

            def ema_retrieve():
                return ema.average(batch_mean), ema.average(batch_var)

            # If the net is being trained, update the average every training step
            def ema_update():
                ema_apply = ema.apply([batch_mean, batch_var])

                # Make sure to compute the new means and variances prior to returning their values
                with tf.control_dependencies([ema_apply]):
                    #identity= Return a tensor with the same shape and contents as input.
                    return tf.identity(batch_mean), tf.identity(batch_var)

            # Retrieve the means and variances and apply the BN transformation
            # cond will return the first function if the condition is true, or the second function if the condition is false
            mean, var = tf.cond(tf.equal(is_training, True), ema_update, ema_retrieve)
            bn_inputs = tf.nn.batch_normalization(inputs, mean, var, offset, scale, EPSILON)

        return bn_inputs

    def _conv_block(self, inputs, maps_shape, stride, name,
                    norm=True, padding='SAME', activation=tf.nn.relu):
        """Convolve inputs and return their batch normalized tensor.

        Args:
            inputs: 4D tensor representing image layer encodings
            maps_shape: list representing the shape of the layer weights
            stride: int representing stride length
            name: string assigned as the tf op names
            norm: bool representing whether or not to normalize layer inputs
            padding: string representing padding type
            activation: tf.nn activation function

        Returns:
            maps: 4D tensor representing convolved feature maps
        """

        with tf.variable_scope(name):
            if name == 'output':
                activation = tf.nn.sigmoid

            filters = self._get_weights(maps_shape)
            filter_maps = tf.nn.conv2d(inputs, filters, [1, stride, stride, 1], padding=padding)
            num_out_maps = maps_shape[3]
            #bias = constant which help the model to fit better to the given data
            bias = tf.Variable(tf.constant(.1, shape=[num_out_maps]))
            #Adds bias to a value
            filter_maps = tf.nn.bias_add(filter_maps, bias)

            if norm:
                filter_maps = tf.layers.BatchNormalization()(filter_maps)
                #filter_maps = self._batch_normalize(filter_maps, maps_shape[3], is_training=True)

            maps = activation(filter_maps) if activation else filter_maps
            return maps

    def _upsample_block(self, inputs, maps_shape, stride, name):
        """Upsamples inputs using transposed convolution.

        Args:
            inputs: 4D tensor representing image layer encodings
            maps_shape: list representing the shape of the layer weights
            stride: int representing stride length
            name: string assigned as the tf op names

        Returns:
            maps: 4D tensor representing upsampled feature maps
        """

        with tf.variable_scope(name):
            filters = self._get_weights(maps_shape)

            # Get dimensions to use for the upsample operator
            batch, height, width, channels = inputs.get_shape().as_list()
            out_height = height * stride
            out_width = width * stride
            out_size = maps_shape[2]
            out_shape = tf.stack([batch, out_height, out_width, out_size])
            stride = [1, stride, stride, 1]

            # Upsample and normalize the biased outputs
            upsample = tf.nn.conv2d_transpose(inputs, filters, output_shape=out_shape, strides=stride)
            bias = tf.Variable(tf.constant(.1, shape=[out_size]))
            upsample = tf.nn.bias_add(upsample, bias)
            #bn_maps = self._batch_normalize(upsample, maps_shape[2], is_training=True)
            bn_maps = tf.layers.BatchNormalization()(upsample)
            maps = tf.nn.relu(bn_maps)

            return maps

    def _residual_block(self, inputs, maps_shape, stride, name):
        """Residual block comprised of two conv layers and aims to add long short-term memory to the network.

        Args:
            inputs: 4D tensor representing image layer encodings
            maps_shape: list representing the shape of the layer weights
            stride: int representing stride length
            name: string assigned as the tf op names

        Returns:
            maps: 4D tensor representing feature maps
        """

        with tf.variable_scope(name):
            conv1 = self._conv_block(inputs, maps_shape, stride=stride, padding='VALID', name='c1')
            conv2 = self._conv_block(conv1, maps_shape, stride=stride, padding='VALID', name='c2', activation=None)

            batch = inputs.get_shape().as_list()[0]
            patch_height, patch_width, num_filters = conv2.get_shape().as_list()[1:]
            #tf.stack = Stacks a list of rank-R tensors into one rank-(R+1) tensor
            out_shape = tf.stack([batch, patch_height, patch_width, num_filters])
            cropped_inputs = tf.slice(inputs, [0, 1, 1, 0], out_shape)
            maps = conv2 + cropped_inputs

            return maps


def load_img_to(path):
    """Returns a numpy array of an image specified by its path.

    Args:
        path: string representing the file path of the image to load

    Returns:
        resized_img: numpy array representing the loaded RGB image
        shape: the image shape
    """

    # Load image [height, width, depth]
    img = skimage.io.imread(path) / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()

    # Crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    shape = list(img.shape)

    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (shape[0], shape[1]))
    return resized_img, shape



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

