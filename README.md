# Parallel Image Processing GUI

A Python-based graphical application that performs **parallel image processing** on a folder of images using multiple algorithms. The app leverages **PyTorch**, **OpenCV**, and **CustomTkinter**, with optional **CUDA (GPU) acceleration**, to efficiently process large image batches.

---

## 🚀 Features

- Modern dark-themed GUI using **CustomTkinter**
- Parallel image processing with `ThreadPoolExecutor`
- Automatic GPU (CUDA) detection using PyTorch
- Multiple image processing algorithms:
  - Grayscale
  - Gaussian Blur
  - Edge Detection (Canny)
  - Histogram Equalization
  - Median Filter
  - Bilateral Filter
  - Median Blur
- Progress bar with real-time status updates
- Automatically organizes processed images into folders by algorithm

---

## 🖼️ Supported Image Formats

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.tiff`

---

## 🧠 Algorithms Used

| Algorithm | Description |
|---------|-------------|
| Grayscale | Converts image to grayscale |
| Blur | Gaussian blur smoothing |
| Edge Detection | Detects edges using Canny |
| Histogram Equalization | Improves contrast |
| Median Filter | Removes salt-and-pepper noise |
| Bilateral Filter | Noise reduction while preserving edges |
| Median Blur | Median-based smoothing |

---

## 📂 Output Structure

Processed_Images/
│
-├── Grayscale/               # All images converted to grayscale
-├── Blur/                    # All images with Gaussian blur applied
-├── Edge Detection/          # All images with Canny edge detection
-├── Histogram Equalization/  # All images with histogram equalization
-├── Median Filter/           # All images processed with median filter
-├── Bilateral Filter/        # All images processed with bilateral filter
-└── Median Blur/             # All images processed with median blur





