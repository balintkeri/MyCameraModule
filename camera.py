from picamzero import Camera
import cv2
import numpy as np
import copy
import time

POSITION_NUMBER = 24


class SandbergCamera():
    def __init__(self):
        self.cam = cv2.VideoCapture(0, cv2.CAP_V4L2)

        # Force YUYV color format instead of MJPG
        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        if not self.cam.isOpened():
            print("❌ Cannot open camera")
            exit()

    def take_photo(self, filename):
        ret, frame = self.cam.read()

        if ret:
            # Save the image
            cv2.imwrite(filename, frame)
            print(f"✅ Image saved as {filename}")
        else:
            print("❌ Failed to capture image")

        self.cam.release()

class CameraHandler:
    def __init__(self):
        try:
            self.cam = Camera()
        except:
            self.cam = SandbergCamera()

    def getPhoto(self, first = True):
        self.cam.take_photo("camera.jpg")
        img = cv2.imread("camera.jpg", cv2.IMREAD_COLOR)
        return img

    def convertToGray(self, img):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray_image

    def convertToBinary(self, img, threshhold = 230):
        ret,thresh1 = cv2.threshold(img,threshhold,255,cv2.THRESH_BINARY)
        return thresh1

    def convertToAdaptiveBinary(self, img):
        thresh1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        return thresh1
    
    def savePhoto(self, img, title = "photo"):
        cv2.imwrite(f"camera_{title}.jpg", img)


    def erode(self, img, time):
        kernel = np.ones((5,5), np.uint8)
        erosion = cv2.erode(img, kernel, iterations = time)
        return erosion
    
    def dilate(self, img, time):
        kernel = np.ones((3,3), np.uint8)
        dilation = cv2.dilate(img, kernel, iterations = time)
        return dilation
    
    def drawBlobs(self, img, binary_img):   
        binary_img = cv2.bitwise_not(binary_img)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle
        return img                      


class CameraAdapter:
    def __init__(self):
        self.camera = CameraHandler()


    def orderElements(self, elements):
        elements.sort(key=lambda x: x[1])

        firstLine = elements[0:3]
        secondLine = elements[3:6]
        thirdLine = elements[6:9]
        fourthLine = elements[9:15]
        fifthLine = elements[15:18]
        sixthLine = elements[18:21]
        seventhLine = elements[21:24]

        firstLine.sort(key=lambda x: x[0])
        secondLine.sort(key=lambda x: x[0])
        thirdLine.sort(key=lambda x: x[0])
        fourthLine.sort(key=lambda x: x[0])
        fifthLine.sort(key=lambda x: x[0])
        sixthLine.sort(key=lambda x: x[0])
        seventhLine.sort(key=lambda x: x[0])

        firstLine = [element[2] for element in firstLine]
        secondLine = [element[2] for element in secondLine]
        thirdLine = [element[2] for element in thirdLine]
        fourthLine = [element[2] for element in fourthLine]
        fifthLine = [element[2] for element in fifthLine]
        sixthLine = [element[2] for element in sixthLine]
        seventhLine = [element[2] for element in seventhLine]


        board = [
            [firstLine, [fourthLine[0], 0, fourthLine[-1]], seventhLine],
            [secondLine,[fourthLine[1], 0, fourthLine[-2]], sixthLine],
            [thirdLine, [fourthLine[2], 0, fourthLine[-3]], fifthLine],
        ]

        return board
        

    def iterateOverPositions(self):
        img = self.camera.getPhoto( first=False)
        img = self.drawRectangle(img, self.positions)

        plotImg = copy.deepcopy(img)

        elements = []

        for contour in self.positions:
            x, y, w, h = cv2.boundingRect(contour)
            mean_intensity = self.getIntenseity(img, contour)
            mean_intensity_orig = self.getIntenseity(self.table, contour)
            
            if (mean_intensity_orig - mean_intensity)/mean_intensity_orig > 0.15:
                pieceType = 2
                cv2.rectangle(plotImg, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw red rectangle
            elif (mean_intensity_orig - mean_intensity)/mean_intensity_orig < -0.06:
                pieceType = 1
                cv2.rectangle(plotImg, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle
            else:
                pieceType = 0
                cv2.rectangle(plotImg, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw blue rectangle

                
            elements.append([x, y,pieceType])

        self.camera.savePhoto(plotImg, title="positions")
                
        return self.orderElements(elements)


    def getIntenseity(self, img, contour):
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        masked_img = cv2.bitwise_and(img, mask)
        gray_masked = self.camera.convertToGray(masked_img)
        mean_intensity = cv2.mean(gray_masked, mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))[0]
        return mean_intensity


    def getTableMask(self, img, threshhold):
        gray = self.camera.convertToGray(img)
        self.camera.savePhoto(gray, title="gray_mask")
        binary = self.camera.convertToBinary(gray, threshhold=threshhold)
        self.camera.savePhoto(binary, title="binary_mask")
        

        dilatated = self.camera.dilate(binary, 50)
        self.camera.savePhoto(dilatated, title="dilatated_mask")
        eroded = self.camera.erode(dilatated, 50)
        self.camera.savePhoto(eroded, title="eroded_mask")
        mask = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
        
        return mask
    
    def getPositions(self, img):
        gray = self.camera.convertToGray(img)
        self.camera.savePhoto(gray, title="gray_table")
        binary = self.camera.convertToBinary(gray)
        self.camera.savePhoto(binary, title="binary_table")
        dilatated = self.camera.dilate(binary, 13)

        self.camera.savePhoto(dilatated, title="dilatated_table")
        eroded = self.camera.erode(dilatated, 13)
        self.camera.savePhoto(eroded, title="eroded_table")
        img = cv2.bitwise_not(eroded)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours

    def mask(self, img,mask, white= True):
        if white:
            img[mask == 0] = 255
        else:
            img[mask == 0] = 0
        return img

    def drawRectangle(self, img, positions):
        for contour in positions:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle
        return img

    def getTable(self):
        self.table = self.camera.getPhoto()
        found = False
        threshhold = 250
        counter = 0

        while not found:
            img = copy.deepcopy(self.table)
            mask = self.getTableMask(img, threshhold)
            img = self.mask(img, mask)
            
            self.camera.savePhoto(img, title="mask")
            positions = self.getPositions(img)

            blob_count = len(positions)
            print(f"Threshold: {threshhold}")
            print(f"Number of blobs: {blob_count}")
            
            img = copy.deepcopy(self.table)
            img = self.drawRectangle(img, positions)
            self.camera.savePhoto(img, title="table")

            if blob_count == POSITION_NUMBER:
                found = True
                self.positions = positions
                print("Found table")
            elif blob_count > POSITION_NUMBER:
                raise Exception("Too many blobs found")
            else:
                if threshhold == 0:
                    print("No table found")
                    raise Exception("No table found")
            
            
            threshhold = threshhold- 1
        


