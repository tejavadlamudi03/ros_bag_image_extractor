# extract_images.py
from rosbags.highlevel import AnyReader
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2

BAG_PATH   = Path('/home/bhanu/Downloads/Dataset')
SHARP_DIR  = Path('/home/bhanu/Downloads/Images/sharp')
BLUR_DIR   = Path('/home/bhanu/Downloads/Images/blurry')
DUP_DIR    = Path('/home/bhanu/Downloads/Images/duplicates')
NO_OBJ_DIR = Path('/home/bhanu/Downloads/Images/no_object')
for d in [SHARP_DIR, BLUR_DIR, DUP_DIR, NO_OBJ_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BLUR_THRESHOLD       = 80.0   # Global sharpness floor
REGION_THRESHOLD     = 100.0   # Per-region sharpness floor
BLUR_REGION_MAX      = 0.5   # If >35% of regions are blurry → reject image
HASH_THRESHOLD       = 10      # Duplicate sensitivity
CONF_THRESHOLD       = 0.4    # YOLO confidence
TARGET_CLASSES       = {2,9}  # car, traffic light

model = YOLO('yolov8m.pt')

def laplacian_var(gray_patch):
    return cv2.Laplacian(gray_patch, cv2.CV_64F).var()

def is_sharp(img, grid=4):
    """
    Global + regional sharpness check.
    Splits image into grid×grid regions, checks what fraction are blurry.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Global check first
    if laplacian_var(gray) < BLUR_THRESHOLD:
        return False

    # Regional check — split into grid×grid patches
    h, w = gray.shape
    rh, rw = h // grid, w // grid
    blurry_regions = 0
    total_regions  = grid * grid

    for i in range(grid):
        for j in range(grid):
            patch = gray[i*rh:(i+1)*rh, j*rw:(j+1)*rw]
            if laplacian_var(patch) < REGION_THRESHOLD:
                blurry_regions += 1

    blurry_ratio = blurry_regions / total_regions
    return blurry_ratio <= BLUR_REGION_MAX  # True = sharp enough

def phash(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    return (resized > resized.mean()).flatten()

def is_duplicate(hash_new, seen_hashes):
    for h in seen_hashes:
        if np.count_nonzero(hash_new != h) <= HASH_THRESHOLD:
            return True
    return False

def contains_target(img):
    results = model(img, verbose=False, conf=CONF_THRESHOLD)[0]
    detected = set(results.boxes.cls.cpu().numpy().astype(int))
    return bool(detected & TARGET_CLASSES)

sharp_count  = 0
blur_count   = 0
dup_count    = 0
no_obj_count = 0
seen_hashes  = []

with AnyReader([BAG_PATH]) as reader:
    for conn, timestamp, rawdata in reader.messages():
        if conn.topic == '/camera/camera/color/image_raw':
            msg = reader.deserialize(rawdata, conn.msgtype)
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Step 1: Duplicate check
            h = phash(img)
            if is_duplicate(h, seen_hashes):
                cv2.imwrite(str(DUP_DIR / f'dup_{dup_count:06d}.png'), img)
                dup_count += 1
                continue
            seen_hashes.append(h)

            # Step 2: Object detection
            if not contains_target(img):
                cv2.imwrite(str(NO_OBJ_DIR / f'no_obj_{no_obj_count:06d}.png'), img)
                no_obj_count += 1
                continue

            # Step 3: Global + regional blur check
            if is_sharp(img, grid=4):
                cv2.imwrite(str(SHARP_DIR / f'frame_{sharp_count:06d}.png'), img)
                sharp_count += 1
            else:
                cv2.imwrite(str(BLUR_DIR / f'frame_{blur_count:06d}.png'), img)
                blur_count += 1

            total = sharp_count + blur_count
            if total % 50 == 0:
                print(f'  Sharp: {sharp_count}  |  Blurry: {blur_count}  |  No object: {no_obj_count}  |  Dups: {dup_count}')

print(f'\n✅ Done!')
print(f'   Sharp      : {sharp_count} images → {SHARP_DIR}')
print(f'   Blurry     : {blur_count} images → {BLUR_DIR}')
print(f'   No object  : {no_obj_count} images → {NO_OBJ_DIR}')
print(f'   Duplicates : {dup_count} images → {DUP_DIR}')