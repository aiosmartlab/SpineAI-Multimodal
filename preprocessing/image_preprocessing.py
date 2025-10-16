import cv2
import numpy as np
from PIL import Image

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def resize_image(self, image):
        """Resize ภาพเป็น 224x224"""
        return cv2.resize(image, self.target_size)
    
    def normalize_pixel_values(self, image):
        """Normalize pixel values ให้อยู่ระหว่าง 0-1"""
        return image.astype(np.float32) / 255.0
    
    def histogram_equalization(self, image):
        """ปรับ contrast ด้วย histogram equalization"""
        if len(image.shape) == 3:
            # แปลงเป็น grayscale
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.equalizeHist(image)
    
    def preprocess(self, image_path):
        """Pipeline ทั้งหมด"""
        image = cv2.imread(image_path)
        image = self.resize_image(image)
        image = self.histogram_equalization(image)
        image = self.normalize_pixel_values(image)
        return image