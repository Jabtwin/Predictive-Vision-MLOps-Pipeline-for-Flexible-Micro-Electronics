import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=(256, 256)):
    """
    Returns an Albumentations composition for training.
    Simulates reality physical noise: reflection, warpage, motion blur.
    """
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        # Simulate factory lighting and metallic reflections
        A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, num_flare_circles_lower=1, num_flare_circles_upper=3, src_radius=150, src_color=(255, 255, 255), p=0.3),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.3),
        
        # Simulate material warpage and bending (crucial for flexible micro-electronics)
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
        
        # Simulate robotic arm motion blur
        A.MotionBlur(blur_limit=7, p=0.3),
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transforms(img_size=(256, 256)):
    """
    Validation transforms (resize and normalize only).
    """
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
