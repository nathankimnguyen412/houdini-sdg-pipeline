"""
Resize evaluation photos to 512x512 for the Lego detector.

Usage:
  python resize_photos.py --input /path/to/raw/photos --output /path/to/resized/

Resizes all .jpg/.jpeg/.png images to 512x512 using center-crop-then-resize
to avoid distorting the aspect ratio too badly. Originals are not modified.
"""

import argparse
import glob
import os
import cv2


def resize_with_pad(img, target_size=512):
    """Resize image to target_size x target_size.
    
    Strategy: resize so the shorter side equals target_size,
    then center-crop the longer side. This avoids heavy distortion
    while filling the full 512x512 frame.
    """
    h, w = img.shape[:2]

    # Scale so shorter side = target_size
    if w < h:
        scale = target_size / w
        new_w = target_size
        new_h = int(h * scale)
    else:
        scale = target_size / h
        new_h = target_size
        new_w = int(w * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center crop to target_size x target_size
    y_start = (new_h - target_size) // 2
    x_start = (new_w - target_size) // 2
    cropped = resized[y_start:y_start + target_size, x_start:x_start + target_size]

    return cropped


def main():
    parser = argparse.ArgumentParser(description="Resize photos for Lego detector evaluation")
    parser.add_argument("--input", required=True, help="Folder with raw phone photos")
    parser.add_argument("--output", required=True, help="Folder to save resized 512x512 images")
    parser.add_argument("--size", type=int, default=512, help="Target size (default: 512)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Find all images
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.input, ext)))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"No images found in {args.input}")
        return

    print(f"Found {len(image_paths)} images in {args.input}")
    print(f"Resizing to {args.size}x{args.size} and saving to {args.output}\n")

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: Could not read {img_path}, skipping")
            continue

        original_size = f"{img.shape[1]}x{img.shape[0]}"
        resized = resize_with_pad(img, target_size=args.size)

        # Save with a clean numbered filename
        out_name = f"eval_{i:03d}.png"
        out_path = os.path.join(args.output, out_name)
        cv2.imwrite(out_path, resized)
        print(f"  {os.path.basename(img_path)} ({original_size}) -> {out_name}")

    print(f"\nDone. {len(image_paths)} images saved to {args.output}")


if __name__ == "__main__":
    main()
