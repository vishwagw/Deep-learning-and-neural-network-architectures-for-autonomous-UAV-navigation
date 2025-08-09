from albumentations import (
    Compose, RandomBrightnessContrast, HueSaturationValue,
    ShiftScaleRotate, HorizontalFlip, CoarseDropout
)

aug = Compose([
    RandomBrightnessContrast(p=0.5),
    HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
    HorizontalFlip(p=0.5),
    CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3)
])

# Apply during training
augmented_frame = aug(image=frame)['image']