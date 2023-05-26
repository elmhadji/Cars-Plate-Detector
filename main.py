from PyQt5.QtWidgets import QApplication ,QMainWindow ,QFileDialog ,QLabel , QPushButton ,QMessageBox
from PyQt5.QtGui import QIcon ,QPixmap ,QImage
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi 
import numpy as np
from cv2 import imread ,cvtColor ,bilateralFilter ,Canny ,contourArea,approxPolyDP ,drawContours,\
      bitwise_and,findContours ,COLOR_BGR2GRAY ,RETR_TREE ,CHAIN_APPROX_SIMPLE
import imutils
import easyocr
class MainPage(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('Resource/main.ui', self)
        self.image = None
        # Define the button and the labels
        self.select_image = self.findChild(QPushButton ,'select_image')
        self.reset = self.findChild(QPushButton ,'reset')
        self.filter = self.findChild(QPushButton ,'apply_filter')
        self.gray_scale = self.findChild(QPushButton ,'transform_gray_scale')
        self.show_result = self.findChild(QPushButton ,'show_result')
        self.image_label = self.findChild(QLabel ,'image')
        self.output_label = self.findChild(QLabel ,'output')

        # Add icon to the select image button
        self.select_image.setIcon(QIcon("Resource/empty-folder.png"))

        # Add functionnality to each button
        self.select_image.clicked.connect(self.open_file)
        self.reset.clicked.connect(self.reset_image)
        self.filter.clicked.connect(self.apply_bilateral_filter)
        self.gray_scale.clicked.connect(self.apply_gray_scale)
        self.show_result.clicked.connect(self.apply_result)


    def open_file(self):
        fileName = QFileDialog.getOpenFileName(self,
                                               ("Open Image"),
                                                 "Picture", 
                                                 ("JPG (*.JPG *.jpg)"))
        if fileName[0]!='':
            self.select_image.setStyleSheet("border:0;background: green;display: block;\
                                           text-align: center;width: 200px;color: white;\
                                           border-radius: 14px;transition: 0.25s;\
                                            font-weight:bold;")
            self.image = imread(fileName[0])
            self.image_label.setPixmap(QPixmap(fileName[0]).scaled(
                700, 450, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                ))

    def reset_image(self):
        if self.image is not None:
            # Convert numpy array to QImage
            qimg = QImage(self.image.data, self.image.shape[1], self.image.shape[0],self.image.strides[0], QImage.Format_BGR888)

            # Convert QImage to QPixmap
            qpixmap = QPixmap.fromImage(qimg)

            # Resize the QPixmap
            desired_width = 700  # Set the desired width of the resized image
            desired_height = 450  # Set the desired height of the resized image
            qpixmap_resized = qpixmap.scaled(
                desired_width, desired_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(qpixmap_resized)
        else:
            dlg = QMessageBox()
            dlg.setWindowTitle("ERROR")
            dlg.setText("Please select an Image ")
            dlg.exec()

    def apply_gray_scale(self):
        if self.image is not None:
            gray_image =cvtColor(self.image,COLOR_BGR2GRAY)

            # Convert numpy array to QImage
            qimg = QImage(gray_image.data, gray_image.shape[1], gray_image.shape[0],gray_image.strides[0], QImage.Format_Grayscale8)

            # Convert QImage to QPixmap
            qpixmap = QPixmap.fromImage(qimg)

            # Resize the QPixmap
            desired_width = 700  # Set the desired width of the resized image
            desired_height = 450  # Set the desired height of the resized image
            qpixmap_resized = qpixmap.scaled(
                desired_width, desired_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(qpixmap_resized)
        else:
            dlg = QMessageBox()
            dlg.setWindowTitle("ERROR")
            dlg.setText("Please select an Image ")
            dlg.exec()

    def apply_bilateral_filter(self):
        if self.image is not None:
            gray_image = cvtColor(self.image,COLOR_BGR2GRAY)
            img_bfilter = bilateralFilter(gray_image ,11,17,17)
            img_edges = Canny(img_bfilter,30,200)

            # Convert numpy array to QImage
            qimg = QImage(img_edges.data, img_edges.shape[1], img_edges.shape[0],img_edges.strides[0], QImage.Format_Grayscale8)

            # Convert QImage to QPixmap
            qpixmap = QPixmap.fromImage(qimg)

            # Resize the QPixmap
            desired_width = 700  # Set the desired width of the resized image
            desired_height = 450  # Set the desired height of the resized image
            qpixmap_resized = qpixmap.scaled(
                desired_width, desired_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(qpixmap_resized)
        else:
            dlg = QMessageBox()
            dlg.setWindowTitle("ERROR")
            dlg.setText("Please select an Image ")
            dlg.exec()
    
    def apply_result(self):
        try:
            if self.image is not None:
                # Grayscale
                gray_image = cvtColor(self.image,COLOR_BGR2GRAY)

                # Apply filter and find edges for localization
                img_bfilter = bilateralFilter(gray_image ,11,17,17)
                img_edges = Canny(img_bfilter,30,200)
                
                # Find Contours and Apply Mask
                keypoints = findContours(img_edges.copy(), RETR_TREE, CHAIN_APPROX_SIMPLE)
                contours = imutils.grab_contours(keypoints)
                contours = sorted(contours, key=contourArea, reverse=True)[:10]

                location = None
                for contour in contours:
                    approx = approxPolyDP(contour, 10, True)
                    if len(approx) == 4:
                        location = approx
                        break

                mask = np.zeros(gray_image.shape, np.uint8)
                new_image = drawContours(mask, [location], 0,255, -1)
                new_image = bitwise_and(self.image, self.image, mask=mask)
                (x,y) = np.where(mask==255)
                (x1, y1) = (np.min(x), np.min(y))
                (x2, y2) = (np.max(x), np.max(y))
                cropped_image = gray_image[x1:x2+1, y1:y2+1]

                # Use Easy OCR To Read Text
                reader = easyocr.Reader(['en'])
                result = reader.readtext(cropped_image)

                #print(result)
                self.output_label.setText(f"The Number Plate is {result[0][-2]}")
            else:
                dlg = QMessageBox()
                dlg.setWindowTitle("ERROR")
                dlg.setText("Please select an Image ")
                dlg.exec()
        except Exception as e:
            dlg = QMessageBox()
            dlg.setWindowTitle("ERROR")
            dlg.setText(f"An error occurred: {str(e)}")
            dlg.exec()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    myapp = MainPage()
    myapp.show()

    try:
        sys.exit(app.exec())
    except SystemExit:
        print('clossing window')