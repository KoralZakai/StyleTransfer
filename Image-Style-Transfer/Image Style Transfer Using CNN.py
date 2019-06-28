"""Image Style Transfer Using Convolutional Neural Network
code Written in python, Ui made with PyQt5"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
import threading
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QWidget, QCheckBox
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QFile, QTextStream
import ctypes

#don't delete using python files with image and css source
import design
import css
import os

# global variables created to control the UI and code parameters.
global content_path
global style_path
global outputImage
global pixmap
global exitflag
exitflag=0
global flagContent
flagContent=0
global flagStyle
flagStyle=0
global flagFinishGenerate
flagFinishGenerate=0
global count
count=0
global iter
iter = 0

"""MainWindowGui is the main class of the UI,
all UI parameters and code functions defined here."""
class MainWindowGui(QWidget):
    def __init__(self, parent=None):
        super(MainWindowGui, self).__init__(parent)
        # init the initial parameters of this GUI
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
        self.title = 'Style Maker'
        self.width = w
        self.height = h

        self.initUI()

    def initUI(self):
        file = QFile(':css/StyleSheet.css')
        file.open(QFile.ReadOnly)
        stream = QTextStream(file)
        text = stream.readAll()
        self.setStyleSheet(text)
        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon(":Pictures/logo.png"))
        self.setGeometry(0, 0, self.width, self.height-60)

        #Creating main container-frame, parent it to QWindow
        self.main_frame = QtWidgets.QFrame(self)
        self.main_frame.setObjectName("MainFrame")
        self.main_frame.setFixedWidth(self.width)
        self.main_frame.setFixedHeight(self.height)
        self.main_layout = QtWidgets.QVBoxLayout(self.main_frame)

        # the Icons sub frame
        self.Iconsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.Iconsub_Frame.setFixedHeight(80)
        self.main_layout.addWidget(self.Iconsub_Frame)
        self.Iconsub_Layout = QtWidgets.QHBoxLayout(self.Iconsub_Frame)
        self.Iconsub_Layout.setAlignment(Qt.AlignLeft)

        # help button
        helpBtn = QtWidgets.QPushButton("", self)
        helpBtn.setStyleSheet("QPushButton {background: url(:Pictures/help.png) no-repeat transparent;} QToolTip {color: black; font-weight: normal;} ")
        helpBtn.setToolTip('Show help pdf')
        helpBtn.setFixedWidth(68)
        helpBtn.setFixedHeight(68)
        helpBtn.clicked.connect(self.showHelpPdf)
        self.Iconsub_Layout.addWidget(helpBtn)

        # the Icon sub frame
        self.Logosub_Frame = QtWidgets.QFrame(self.main_frame)
        self.Logosub_Frame.setFixedWidth(self.width)
        self.main_layout.addWidget(self.Logosub_Frame)
        self.Logosub_Layout = QtWidgets.QHBoxLayout(self.Logosub_Frame)
        self.Logosub_Layout.setAlignment(Qt.AlignCenter)

        # Setting up the logo
        logo = QtWidgets.QLabel('', self)
        pixmap = QPixmap(":Pictures/logo.png")
        pixmap = pixmap.scaled(260, 260)
        logo.setPixmap(pixmap)
        self.Logosub_Layout.addWidget(logo)
        logo.setAlignment(Qt.AlignCenter)

        # The Button sub frame
        self.Buttonsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.Buttonsub_Frame)
        self.Buttonsub_Layout = QtWidgets.QHBoxLayout(self.Buttonsub_Frame)
        self.Buttonsub_Frame.setFixedWidth(self.width)
        self.Buttonsub_Layout.setAlignment(Qt.AlignTop | Qt.AlignCenter)

        # start to create style button
        StartCreateNewBtn = QtWidgets.QPushButton("Style your image", self)
        StartCreateNewBtn.setObjectName("MainGuiButtons")
        StartCreateNewBtn.clicked.connect(self.openTransferImageGui)
        self.Buttonsub_Layout.addWidget(StartCreateNewBtn)

        # Footer layout
        creditsLbl = QtWidgets.QLabel('Created By Koral Zakai & May Steinfeld, '
                                      'Supervisor: Zeev Vladimir Volkovich, '
                                      '03/06/2019')
        creditsLbl.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(creditsLbl)

        # show the window
        self.showMaximized()

    def openTransferImageGui(self):
        transferImage = TransferImageGui(self)
        transferImage.show()
        self.main_frame.setVisible(False)

    def showHelpPdf(self):
        import os
        filename = 'Help.pdf'
        try:
            os.startfile(filename)
        except:
            return

class TransferImageGui(QWidget):
    def __init__(self, parent=None):
        super(TransferImageGui, self).__init__(parent)

        # init the initial parameters of this GUI
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
        self.title = 'Style Maker'
        self.width = w
        self.height = h
        self.initUI2()

    def initUI2(self):
        global flagContent
        flagContent= 0
        global flagStyle
        flagStyle= 0

        file = QFile(':css/StyleSheet.css')
        file.open(QFile.ReadOnly)
        stream = QTextStream(file)
        text = stream.readAll()
        self.setStyleSheet(text)
        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon(":Pictures/logo.png"))
        self.setGeometry(0, 0, self.width, self.height - 60)

        # Creating main container-frame, parent it to QWindow
        self.main_frame = QtWidgets.QFrame(self)
        self.main_frame.setObjectName("MainFrame")
        self.main_frame.setFixedWidth(self.width)
        self.main_frame.setFixedHeight(self.height)
        # the first sub window
        self.main_layout = QtWidgets.QVBoxLayout(self.main_frame)

        # home and help buttons
        # the Icons sub frame
        self.Iconsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.Iconsub_Frame.setFixedHeight(80)
        self.main_layout.addWidget(self.Iconsub_Frame)
        self.Iconsub_Layout = QtWidgets.QHBoxLayout(self.Iconsub_Frame)
        self.Iconsub_Layout.setAlignment(Qt.AlignLeft)

        # help button
        helpBtn = QtWidgets.QPushButton("", self)
        helpBtn.setStyleSheet("QPushButton {background: url(:Pictures/help.png) no-repeat transparent;} QToolTip {color: black; font-weight: normal;} ")
        helpBtn.setToolTip('Show help pdf')
        helpBtn.setFixedWidth(68)
        helpBtn.setFixedHeight(68)
        helpBtn.clicked.connect(MainWindowGui.showHelpPdf)
        self.Iconsub_Layout.addWidget(helpBtn)

        # home button
        homeBtn = QtWidgets.QPushButton("", self)
        homeBtn.setStyleSheet("QPushButton {background: url(:Pictures/home.png) no-repeat transparent;} ")
        homeBtn.setFixedWidth(68)
        homeBtn.setFixedHeight(68)
        homeBtn.clicked.connect(self.showHome)
        self.Iconsub_Layout.addWidget(homeBtn)

        self.buttonsSub_Frame = QtWidgets.QFrame(self.main_frame)
        self.buttonsSub_Frame.setFixedWidth(self.width)
        #self.buttonsSub_Frame.setFixedHeight(200)
        self.main_layout.addWidget(self.buttonsSub_Frame)
        self.buttonsSub_Layout = QtWidgets.QHBoxLayout(self.buttonsSub_Frame)
        self.buttonsSub_Layout.setAlignment(Qt.AlignCenter|Qt.AlignTop)

        QtCore.QMetaObject.connectSlotsByName(main)

        # upload content button
        contentBtn = QtWidgets.QPushButton("Upload content image", self)
        contentBtn.setObjectName("MainGuiButtons")
        contentBtn.clicked.connect(self.setContentImage)
        self.buttonsSub_Layout.addWidget(contentBtn)
        #self.contentframe.show()

        # upload style
        StyleBtn = QtWidgets.QPushButton("Upload style Image", self)
        StyleBtn.setObjectName("MainGuiButtons")
        StyleBtn.clicked.connect(self.setStyleImage)
        self.buttonsSub_Layout.addWidget(StyleBtn)
        #self.styleframe.show()

        # The photos sub frame
        self.photosSubframe = QtWidgets.QFrame(self.main_frame)
        self.photosSubframe.setFixedWidth(self.width)
        self.photosSubframe.setFixedHeight(10)
        self.main_layout.addWidget(self.photosSubframe)
        self.photosSub_Layout = QtWidgets.QHBoxLayout(self.photosSubframe)
        self.photosSub_Layout.setAlignment(Qt.AlignCenter)

        self.contentframe = QtWidgets.QLabel(self.main_frame)
        self.contentframe.setGeometry(QtCore.QRect(self.width * 3.2 / 10, self.height * 1.7 / 7, 251, 191))
        self.contentframe.setFrameShape(QtWidgets.QFrame.NoFrame)
        pixmap = QPixmap(":Pictures/imageNeedUpload.png")
        pixmap = pixmap.scaled(256, 256)
        self.contentframe.setPixmap(pixmap)
        self.contentframe.setText("")
        self.contentframe.setScaledContents(True)
        self.contentframe.setObjectName("contentframe")
        # self.contentframe.hide()

        self.styleframe = QtWidgets.QLabel(self.main_frame)
        self.styleframe.setGeometry(QtCore.QRect(self.width * 5.1 / 10, self.height * 1.7 / 7, 251, 191))
        self.styleframe.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.styleframe.setPixmap(pixmap)
        self.styleframe.setText("")
        self.styleframe.setScaledContents(True)
        self.styleframe.setObjectName("styleframe")
        # self.styleframe.hide()

        self.details_Frame = QtWidgets.QFrame(self.main_frame)
        self.details_Frame.setFixedWidth(self.width)
        self.main_layout.addWidget(self.details_Frame)
        self.details_Layout = QtWidgets.QHBoxLayout(self.details_Frame)
        self.details_Layout.setAlignment(Qt.AlignCenter)

        iterText = QtWidgets.QLabel('Image quality:')
        self.details_Layout.addWidget(iterText)
        self.iterationbox = QtWidgets.QComboBox(self.main_frame)
        self.iterationbox.addItem("Low")
        self.iterationbox.addItem("Medium")
        self.iterationbox.addItem("High")
        self.details_Layout.addWidget(self.iterationbox)

        resText = QtWidgets.QLabel('    Image resolution:')
        self.details_Layout.addWidget(resText)
        self.resolutionbox = QtWidgets.QComboBox(self.main_frame)
        self.resolutionbox.addItem("Small- 256 px")
        self.resolutionbox.addItem("Medium- 512 px")
        self.resolutionbox.addItem("Large- 1024 px")
        self.details_Layout.addWidget(self.resolutionbox)

        modelText = QtWidgets.QLabel('  Choose model:')
        self.details_Layout.addWidget(modelText)
        self.modelBox = QtWidgets.QComboBox(self.main_frame)
        self.modelBox.addItem("Vgg16")
        self.modelBox.addItem("Vgg19")
        self.details_Layout.addWidget(self.modelBox)

        self.generateBtnSub_Frame = QtWidgets.QFrame(self.main_frame)
        self.generateBtnSub_Frame.setFixedWidth(self.width)
        self.main_layout.addWidget(self.generateBtnSub_Frame)
        self.generateBtnSub_Layout = QtWidgets.QHBoxLayout(self.generateBtnSub_Frame)
        self.generateBtnSub_Layout.setAlignment(Qt.AlignCenter)

        self.generateBtn = QtWidgets.QPushButton("generate", self)
        self.generateBtn.setToolTip('You must upload content and style images first.')
        self.generateBtn.setObjectName("MainGuiButtons")
        self.generateBtn.clicked.connect(self.lunch_thread)
        self.generateBtnSub_Layout.addWidget(self.generateBtn)
        self.generateBtn.setEnabled(True)

        # show the window
        self.showMaximized()

    """lunch_thread control the start of the second thread that running the MainFunc- StyleMakerFunc."""
    def lunch_thread(self):
        if flagStyle == 1 and flagContent == 1:
            outputWindow = OutputImageGui(self)
            outputWindow.getComboBoxValues(self.iterationbox.currentText(), self.resolutionbox.currentText() , self.modelBox.currentText())
            t = threading.Thread(target=outputWindow.Generate)
            t.start()
            outputWindow.show()
            self.main_frame.setVisible(False)


    # Opens home window
    def showHome(self):
        """
        close current window and return to home page
        """
        home = MainWindowGui(self)
        home.show()
        self.main_frame.setVisible(False)

    """setContentImage function control on choosing the content image."""
    def setContentImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileNames(None, "Select Image", "",
                                                             "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fileName:
            global content_path
            content_path = fileName[0]
            pixmap = QtGui.QPixmap(fileName[0])
            pixmap = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio)
            self.contentframe.setPixmap(pixmap)
            self.contentframe.setAlignment(QtCore.Qt.AlignCenter)
            global flagContent
            flagContent = 1
            global flagStyle
            if (flagContent == 1 and flagStyle == 1):
                self.generateBtn.setToolTip(None)

    """setStyleImage function control on choosing the style image."""
    def setStyleImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileNames(None, "Select Image", "",
                                                             "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fileName:
            global style_path
            style_path = fileName[0]
            pixmap = QtGui.QPixmap(fileName[0])
            pixmap = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio)
            self.styleframe.setPixmap(pixmap)
            self.styleframe.setAlignment(QtCore.Qt.AlignCenter)
            global flagStyle
            flagStyle = 1
            global flagContent
            if (flagStyle == 1 and flagContent == 1):
                self.generateBtn.setToolTip(None)

class OutputImageGui(QWidget):
    def __init__(self , parent=None):
        super(OutputImageGui, self).__init__(parent)
        self.show
        # init the initial parameters of this GUI
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
        self.title = 'Style Maker'
        self.width = w
        self.height = h
        self.initUI()

    def initUI(self):
        file = QFile(':css/StyleSheet.css')
        file.open(QFile.ReadOnly)
        stream = QTextStream(file)
        text = stream.readAll()
        self.setStyleSheet(text)
        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon(":Pictures/logo.png"))
        self.setGeometry(0, 0, self.width, self.height - 60)

        # Creating main container-frame, parent it to QWindow
        self.main_frame = QtWidgets.QFrame(self)
        self.main_frame.setObjectName("MainFrame")
        self.main_frame.setFixedWidth(self.width)
        self.main_frame.setFixedHeight(self.height)
        self.main_layout = QtWidgets.QVBoxLayout(self.main_frame)

        # the Icons sub frame
        self.Iconsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.Iconsub_Frame.setFixedHeight(80)
        self.main_layout.addWidget(self.Iconsub_Frame)
        self.Iconsub_Layout = QtWidgets.QHBoxLayout(self.Iconsub_Frame)
        self.Iconsub_Layout.setAlignment(Qt.AlignLeft)

        # help button
        helpBtn = QtWidgets.QPushButton("", self)
        helpBtn.setStyleSheet("QPushButton {background: url(:Pictures/help.png) no-repeat transparent;} QToolTip {color: black; font-weight: normal;} ")
        helpBtn.setToolTip('Show help pdf')
        helpBtn.setFixedWidth(68)
        helpBtn.setFixedHeight(68)
        helpBtn.clicked.connect(MainWindowGui.showHelpPdf)
        self.Iconsub_Layout.addWidget(helpBtn)

        # home button
        self.homeBtn = QtWidgets.QPushButton("", self)
        self.homeBtn.setStyleSheet("QPushButton {background: url(:Pictures/home.png) no-repeat transparent;} ")
        self.homeBtn.setFixedWidth(68)
        self.homeBtn.setFixedHeight(68)
        self.homeBtn.clicked.connect(self.showHome)
        self.Iconsub_Layout.addWidget(self.homeBtn)
        self.homeBtn.setEnabled(False)
        self.homeBtn.setToolTip('Still in generate process')

        # The Button save + output image sub frame
        self.Buttonsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.Buttonsub_Frame)
        self.Buttonsub_Layout = QtWidgets.QVBoxLayout(self.Buttonsub_Frame)
        self.Buttonsub_Frame.setFixedWidth(self.width)
        self.Buttonsub_Layout.setAlignment(Qt.AlignTop | Qt.AlignCenter)

        # Save button for the output image
        self.savebutton = QtWidgets.QPushButton("Save your image", self)
        self.savebutton.setObjectName("MainGuiButtons")
        self.savebutton.clicked.connect(self.saveimage)
        self.Buttonsub_Layout.addWidget(self.savebutton)

        self.outputframe = QtWidgets.QLabel(self.main_frame)
        #self.outputframe.setGeometry(QtCore.QRect(850, 300, 251, 191))
        self.outputframe.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.outputframe.setText("")
        pixmap = QPixmap(":Pictures/gift.png")
        pixmap = pixmap.scaled(256, 256)
        self.outputframe.setPixmap(pixmap)
        self.outputframe.setScaledContents(True)
        self.outputframe.setObjectName("outputframe")
        self.outputframe.setAlignment(Qt.AlignCenter)
        self.Buttonsub_Layout.addWidget(self.outputframe)
        #self.outputframe.hide()

        # The progressBar sub frame
        self.progressBarsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.progressBarsub_Frame)
        self.progressBarsub_Frame.setFixedWidth(self.width)
        self.progressBarsub_Layout = QtWidgets.QHBoxLayout(self.progressBarsub_Frame)
        self.progressBarsub_Layout.setAlignment(Qt.AlignCenter)

        self.progressBar = QtWidgets.QProgressBar(self.main_frame)
        #self.progressBar.setGeometry(QtCore.QRect(0, 0, 256, 31))
        self.progressBar.setFixedWidth(self.width/3)
        self.progressBar.setProperty("value",0)
        self.progressBar.setMaximum(100)
        self.progressBar.setAlignment(Qt.AlignCenter)
        self.progressBar.setObjectName("progressBar")
        self.progressBarsub_Layout.addWidget(self.progressBar)
        #self.progressBar.hide()

        # show the window
        self.showMaximized()

    def getComboBoxValues(self, iterString, resString, modelString):
        self.comboString = iterString
        self.resolutionString  = resString
        self.modelString = modelString

    """saveimage function control the saving of the output image."""
    def saveimage(self):
        global outputImage
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Select Image", "",
                                                            "Image Files (*.jpg *.png *.jpeg *.bmp)")
        if (fileName):
            outputImage.save(fileName)

    # Opens home window
    def showHome(self):
        """
        close current window and return to home page
        """
        home = MainWindowGui(self)
        home.show()
        self.main_frame.setVisible(False)

    """onCountChanged function control on updating the progrssBar."""
    def onCountChanged(self, value):
        #if(value==100):
        #   self.homeBtn.setEnabled(True)
        self.progressBar.setValue(value)

    """Generate function is start when the Generate button pushed. it start the main algorithm."""
    def Generate(self):
        global outputImage
        global exitflag
        exitflag=1
        global flagContent
        global flagStyle
        #if (flagContent == 0 or flagStyle == 0):
            #self.warninglabel.show()
            #return
        #self.actionHome.setEnabled(False)
        #self.actionCreate_New.setEnabled(False)
        #self.actionAbout.setEnabled(False)

        #self.outputframe.setPixmap(QtGui.QPixmap(":Pictures/gift.png"))
        #self.outputframe.show()

        #self.savebutton.raise_()
        self.savebutton.hide()

        #self.progressBar.raise_()
        #self.outputframe.raise_()

        #self.progressBar.setValue(0)
        #self.progressBar.show()

        # iter control the number of iteration the algorithm run, the user choose it.
        global iter
        iter=0
        if self.comboString == 'Low':
            iter=100
        elif self.comboString == 'Medium':
            iter=500
        else:
            iter=1000

        # resulotion control the output image resulotion, the user choose it.
        resolution = 0
        if self.resolutionString == 'Small- 256 px':
            resolution = 256
        elif self.resolutionString == 'Medium- 512 px':
            resolution = 512
        elif self.resolutionString  == 'Large- 1024 px':
            resolution = 1024

        global modelType
        if self.modelString == 'Vgg16':
            modelType = 16
        elif self.modelString == 'Vgg19':
            modelType = 19

        # outputImage get the result from the StyleMakerFunc.
        outputImage = self.StyleMakerFunc(content_path, style_path, iter, resolution, modelType)
        pixmap = QtGui.QPixmap(outputImage.toqpixmap())
        pixmap = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio)
        self.outputframe.setPixmap(pixmap)
        self.outputframe.setAlignment(QtCore.Qt.AlignCenter)
        self.outputframe.show()
        self.savebutton.show()

        global flagFinishGenerate
        flagFinishGenerate = 1
        self.homeBtn.setEnabled(True)
        self.homeBtn.setToolTip(None)

    """exit function control on exit the application."""
    def exit(self):
        if(exitflag == 1):
            self.exit()
        else:
            exit(1)


    """StyleMakerFunc is the main function that running the main algorithm"""
    def StyleMakerFunc(self, content_path, style_path, iter, resolution, modelType):
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

        # load_and_process_img is charge on load the image into the vgg network.
        def load_and_process_img(path_to_img):
            img = load_img(path_to_img)
            if modelType == 16:
                img = tf.keras.applications.vgg16.preprocess_input(img)
            elif modelType == 19:
                img = tf.keras.applications.vgg19.preprocess_input(img)
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
            if modelType == 16:
                vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
            elif modelType == 19:
                vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
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
            return tf.reduce_mean(0.5*tf.square(base_content - target))

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
            # Load our images into the VGG Network
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
            init_image = load_and_process_img(content_path)
            init_image = tfe.Variable(init_image, dtype=tf.float32)
            # We  use Adam Optimizer
            opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

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

def myExitHandler(self):
    exit(1)

"""External class control the thread running the ProgressBar."""
class External(QThread):
    countChanged = pyqtSignal(int)

    def run(self):
        global count
        global iter
        progressVal =((count + 1) / iter) * 100
        self.countChanged.emit(progressVal)
    #exit(1)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindowGui()
    app.aboutToQuit.connect(myExitHandler)  # myExitHandler is a callable
    sys.exit(app.exec_())
