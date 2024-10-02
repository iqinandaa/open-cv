import cv2
import numpy as np
from matplotlib import pyplot as plt

# Fungsi untuk mendeteksi warna buah mangga menggunakan metode Sobel
def detect_mango_color(image_path):
    # Load gambar
    img = cv2.imread(image_path)
    if img is None:
        print("Gambar tidak dapat dimuat. Pastikan path benar.")
        return

    # Simpan salinan asli gambar untuk ditampilkan nantinya
    original = img.copy()

    # Konversi warna BGR ke HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Definisikan batas-batas warna untuk buah mangga (dalam format HSV)
    lower_color = np.array([25, 50, 50])    # Ambil nilai ini berdasarkan analisis warna buah mangga
    upper_color = np.array([35, 255, 255])  # Ambil nilai ini berdasarkan analisis warna buah mangga

    # Membuat mask dari gambar HSV menggunakan batas warna
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Aplikasikan operator Sobel pada kanal Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Menyimpan hasil deteksi tepi dan warna buah mangga
    edges = magnitude.astype(np.uint8)
    detected_color = cv2.bitwise_and(mask, mask, mask=edges)

    # Menggunakan matplotlib untuk menampilkan gambar
    plt.figure(figsize=(10, 10))
    
    plt.subplot(1, 3, 1)
    plt.title('Gambar Asli')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Deteksi Tepi (Sobel)')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Deteksi Warna (HSV)')
    plt.imshow(detected_color, cmap='gray')
    plt.axis('off')
    
    plt.show()

# Contoh penggunaan dengan path gambar dari dataset yang diunduh manual
image_path = 'images.mango/subdataset1/images/mango_01.jpg'  # Sesuaikan dengan path gambar dalam dataset
detect_mango_color(image_path)
