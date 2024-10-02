import cv2
import numpy as np

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
    edges = cv2.convertScaleAbs(magnitude)  # Konversi ke tipe data uint8
    detected_color = cv2.bitwise_and(mask, mask, mask=edges)

    # Menampilkan gambar asli, hasil deteksi tepi, dan hasil deteksi warna
    cv2.imshow('Gambar Asli', original)
    cv2.imshow('Deteksi Tepi (Sobel)', edges)
    cv2.imshow('Deteksi Warna (HSV)', detected_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Contoh penggunaan
if __name__ == "__main__":
    image_path = 'C:\\Users\\Yuda\\Desktop\\Opencv apps\\healthy_003.jpg'  # Ganti dengan path gambar buah mangga Anda
    detect_mango_color(image_path)
