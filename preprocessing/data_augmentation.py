import albumentations as A
from albumentations.pytorch import ToTensorV2

class DataAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.Rotate(limit=15, p=0.5),  # Random rotation -15 to +15 degrees
            A.RandomScale(scale_limit=0.1, p=0.5),  # Scale 0.9 to 1.1
            A.ShiftScaleRotate(
                shift_limit=0.1,  # 10% shift
                scale_limit=0,
                rotate_limit=0,
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),  # 50% horizontal flip
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Gaussian noise
            ToTensorV2()
        ])
    
    def augment(self, image):
        return self.transform(image=image)['image']