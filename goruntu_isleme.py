import tkinter as tk
import cv2
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk  
from tkinter import simpledialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dijital Görüntü İşleme Uygulamaları")
        self.root.geometry('800x600')
        self.root.configure(background='#f0f0f0')

        self.create_widgets()
        self.create_menu()

    def create_widgets(self):
        header_font = ("Verdana", 20, "bold")
        text_font = ("Arial", 14)

        self.header_label = tk.Label(self.root, text="Dijital Görüntü İşleme Dersi", font=header_font, bg='#f0f0f0')
        self.header_label.pack(pady=20)

        self.student_info_label = tk.Label(self.root, text="Öğrenci Numarası: 211229045\nAdınız: ZEYNEP KARAOĞLU", font=text_font, bg='#f0f0f0')
        self.student_info_label.pack(pady=10)

    def create_menu(self):
        
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        homework_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Ödevler", menu=homework_menu)
        homework_menu.add_command(label="Ödev 1: Temel İşlevselliği Oluştur", command=self.setup_homework1)
        homework_menu.add_command(label="Ödev 2: Temel İşlevleri ve Interpolasyon", command=self.setup_homework2)
        homework_menu.add_separator()  # Ayırıcı ekleyelim

        # Yeni alt menü 
        midterm_menu = tk.Menu(homework_menu, tearoff=0)
        homework_menu.add_cascade(label="Vize Ödevi", menu=midterm_menu)
        
        # Soru 1
        midterm_menu.add_command(label="Soru 1", command=self.setup_midterm_question1)
        # Soru 2
        midterm_question2_menu = tk.Menu(midterm_menu, tearoff=0)
        midterm_menu.add_cascade(label="Soru 2", menu=midterm_question2_menu)
   
        midterm_question2_menu.add_command(label="A Kısmı", command=self.setup_midterm_question2_part_a)
        midterm_question2_menu.add_command(label="B Kısmı", command=self.setup_midterm_question2_part_b)
        # Soru 3
        midterm_menu.add_command(label="Soru 3", command=self.setup_midterm_question3)
        # Soru 4
        midterm_menu.add_command(label="Soru 4", command=self.setup_midterm_question4)

    def upload_and_process_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        print(f"Seçilen dosya: {file_path}")

        self.image = Image.open(file_path)  # Görüntüyü self.image'de saklayın
        self.image_for_display = self.image.resize((400, 400), Image.ANTIALIAS)  # Görüntüyü göstermek için boyutlandır
        photo = ImageTk.PhotoImage(self.image_for_display)

        if hasattr(self, 'image_label'):
            self.image_label.pack_forget()

        self.image_label = tk.Label(self.root, image=photo)
        self.image_label.image = photo
        self.image_label.pack(pady=20)

        messagebox.showinfo("İşlem Tamamlandı", "Görüntü başarıyla yüklendi ve işlendi.")
    
    def hide_student_info(self):
        self.student_info_label.pack_forget()
        self.header_label.pack_forget()

    def setup_homework1(self):
        self.hide_student_info()
        
        hw1_title = tk.Label(self.root, text="Ödev 1: Temel İşlevselliği Oluştur", font=("Arial", 16, "bold"), bg='#f0f0f0')
        hw1_title.pack(pady=(10, 0))
        
        upload_button = tk.Button(self.root, text="Görüntü Yükle ve İşle", command=self.upload_and_process_image)
        upload_button.pack(pady=20)
        
    def setup_homework2(self):
        self.hide_student_info()
        hw1_title = tk.Label(self.root, text="Ödev 2: Temel Görüntü Operasyonları ve İnterpolasyon", font=("Arial", 16, "bold"), bg='#f0f0f0')
        hw1_title.pack(pady=(10, 0))
       
        upload_button = tk.Button(self.root, text="Görüntü Yükle", command=self.upload_and_process_image)
        upload_button.pack(pady=20)
        buyut_button = tk.Button(self.root, text="Görüntüyü Büyüt", command=self.buyut)
        buyut_button.pack()
        kucult_button = tk.Button(self.root, text="Görüntüyü Küçült", command=self.kucult)
        kucult_button.pack()
        rotate_button = tk.Button(self.root, text="Görüntüyü Döndür", command=self.rotate_image)
        rotate_button.pack()
        
        self.zoom_in_button = tk.Button(self.root, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.pack(pady=(10,0))

        self.zoom_out_button = tk.Button(self.root, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.pack(pady=(5,10))
    
    def setup_midterm_question1(self):
        
        def sigmoid(x, k=10, x0=0.5):
            """ Standart Sigmoid Fonksiyonu """
            return 1 / (1 + np.exp(-k * (x - x0)))

        def shifted_sigmoid(x, shift=0.5, k=10):
            """ Yatay Kaydırılmış Sigmoid Fonksiyonu """
            return 1 / (1 + np.exp(-k * (x - shift)))

        def sloped_sigmoid(x, k=15):
            """ Eğimli Sigmoid Fonksiyonu """
            return 1 / (1 + np.exp(-k * (x - 0.5)))

        def dual_sigmoid(x, cutoff=0.5, gain=10):
            """ Çift Taraflı Sigmoid Fonksiyonu """
            dark_part = 1 / (1 + np.exp(gain * (cutoff - x)))
            light_part = 1 / (1 + np.exp(-gain * (x - (1 - cutoff))))
            combined = dark_part * light_part
            return combined / np.max(combined)  

        def apply_s_curve(image, func):
            """ S-Curve uygulama fonksiyonu """
            img_normalized = image / 255.0  # Görüntüyü [0, 1] aralığına normalize edin
            img_transformed = func(img_normalized)
            img_scaled_back = np.clip(img_transformed * 255, 0, 255).astype(np.uint8)  # [0, 255] aralığına geri dönüştür
            return img_scaled_back
    # Görüntüyü yükleyin ve griye dönüştürün
        img = Image.open('gorsel1.png').convert('L')  # Görüntü yolunu buraya girin
        img_array = np.array(img)

    # Fonksiyonları uygulayın
        standard_output = apply_s_curve(img_array, lambda x: sigmoid(x))
        shifted_output = apply_s_curve(img_array, lambda x: shifted_sigmoid(x))
        sloped_output = apply_s_curve(img_array, lambda x: sloped_sigmoid(x))
        dual_output = apply_s_curve(img_array, lambda x: dual_sigmoid(x))
        
        result_window = tk.Toplevel(self.root)
        result_window.title("Midterm Question 1 Results")

    # Sonuçları göster
        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        ax[0, 0].imshow(img_array, cmap='gray')
        ax[0, 0].set_title('Original Image')
        ax[0, 1].imshow(standard_output, cmap='gray')
        ax[0, 1].set_title('Standard Sigmoid')
        ax[0, 2].imshow(shifted_output, cmap='gray')
        ax[0, 2].set_title('Shifted Sigmoid')
        ax[1, 0].imshow(sloped_output, cmap='gray')
        ax[1, 0].set_title('Sloped Sigmoid')
        ax[1, 1].imshow(dual_output, cmap='gray')
        ax[1, 1].set_title('Dual Sigmoid')

        for a in ax.flat:
            a.axis('off')
            
        canvas = FigureCanvasTkAgg(fig, master=result_window)
        canvas.draw()
        canvas.get_tk_widget().pack()
        plt.show()
            

    def setup_midterm_question2(self):
        self.setup_midterm_question2_part_a()
        self.setup_midterm_question2_part_b()

    def setup_midterm_question2_part_a(self):
        
        def region_of_interest(img, vertices):
            mask = np.zeros_like(img)
            cv2.fillPoly(mask, vertices, 255)
            masked_image = cv2.bitwise_and(img, mask)
            return masked_image

        def detect_lines(image_path):
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: The image at {image_path} could not be loaded.")
                return

        # Resmi griye çevir ve gürültüyü azalt
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            # Yolun olduğu bölgeyi belirle
            height, width = image.shape[:2]
            region = np.array([
            [(0, height), (width / 2, height / 2), (width, height)]
            ], dtype=np.int32)

            # Yalnızca yolun olduğu bölgeyi işle
            targeted_edges = region_of_interest(edges, region)

            # Hough çizgi tespiti
            lines = cv2.HoughLinesP(targeted_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
    
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            resized_image = cv2.resize(image, (500,400))

            # Sonuçları göster
            cv2.imshow('Detected Lines', resized_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Fonksiyonu çağır
        detect_lines('road_image.jpg')

    

    def setup_midterm_question2_part_b(self):
        def detect_eyes_and_pupils(image_path):
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (ex, ey, ew, eh) in eyes:
                eye_roi = gray[ey:ey+eh, ex:ex+ew]
                # Gözün etrafına daha uygun çapta çember çiz (Yeşil renk kullanarak)
                eye_center = (ex + ew // 2, ey + eh // 2)
                eye_radius = int((ew + eh) / 10)  # Yarıçapı gözün boyutlarına daha uygun hale getir
                cv2.circle(img, eye_center, eye_radius, (0, 255, 0), 2)  # Yeşil çember

                # Göz bebeklerini tespit et
                circles = cv2.HoughCircles(eye_roi, cv2.HOUGH_GRADIENT, dp=1, minDist=ew//4, param1=250, param2=30, minRadius=4, maxRadius=20)
                
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        pupil_center = (i[0] + ex, i[1] + ey)
                        pupil_radius = i[2]
                        cv2.circle(img, pupil_center, pupil_radius, (255, 0, 0), 2)  # Mavi çember
                        
            resized_image = cv2.resize(img, (500,300))

            cv2.imshow('Goz Tespiti',  resized_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        detect_eyes_and_pupils('goz.jpg')  

   

    def setup_midterm_question3(self):
        # Load the image
        image_path = 'araba.png'
        img = cv2.imread(image_path)

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply Gaussian Blur
        gaussian_blur = cv2.GaussianBlur(img_rgb, (5,5), 0)

        # Create a sharpening filter
        sharpening_filter = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gaussian_blur, -1, sharpening_filter)

        # Display the images
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(sharpened)
        plt.title('Deblurred Image')
        plt.axis('off')

        plt.show()

    def setup_midterm_question4(self):
        def load_image(image_path):
            """Görüntüyü diskten yükler ve None kontrolü yapar."""
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f'Görüntü {image_path} konumunda bulunamadı.')
            return image

        def create_color_mask(image, lower_bound, upper_bound):
            """Belirtilen renk aralığına göre bir renk maskesi oluşturur."""
            return cv2.inRange(image, lower_bound, upper_bound)

        def extract_contours(color_mask):
            """Maske üzerinden konturları çıkarır."""
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contours

        def calculate_contour_properties(contours, color_mask):
            """Konturların çeşitli özelliklerini hesaplar ve bir liste döndürür."""
            contour_properties = []
            for index, contour in enumerate(contours):
                x, y, width, height = cv2.boundingRect(contour)
                center_position = (x + width // 2, y + height // 2)
                diagonal_length = int((height**2 + width**2)**0.5)
                contour_area = cv2.contourArea(contour)
                contour_region = color_mask[y:y+height, x:x+width]
                pixel_histogram = cv2.calcHist([contour_region], [0], None, [256], [0,256])
                pixel_histogram /= pixel_histogram.sum()
                contour_entropy = -np.sum([probability * np.log2(probability) for probability in pixel_histogram if probability > 0])
                pixel_mean = np.mean(contour_region)
                pixel_median = np.median(contour_region)

                contour_info = {
                    'Contour Number': index+1, 'Center Position': center_position,
                    'Contour Length': f'{height} px', 'Contour Width': f'{width} px',
                    'Diagonal Length': f'{diagonal_length} px', 'Contour Area': contour_area,
                    'Entropy': contour_entropy, 'Mean Pixel Value': pixel_mean, 'Median Pixel Value': pixel_median
                }
                contour_properties.append(contour_info)
            return contour_properties

        def save_properties_to_excel(contour_data, filename):
            """Verilen veriyi bir Excel dosyasına kaydeder."""
            dataframe = pd.DataFrame(contour_data)
            dataframe.to_excel(filename, index=False)

        def select_color_and_process(image_path):
            """Renk seçimi yapar ve görüntü işleme sürecini başlatır."""
            image = load_image(image_path)
            print("Lütfen açılan resim penceresinden bir renge tıklayın.")
            
            # Görüntüyü göster
            cv2.namedWindow('Image')
            selected_color = []

            def mouse_click(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    colors = image[y, x]
                    selected_color.append(colors)
                    print("Renk Secildi (BGR):", colors)
                    cv2.destroyAllWindows()

            cv2.setMouseCallback('Image', mouse_click)
            cv2.imshow('Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if selected_color:
                # Renk değerlerini kullanarak renk aralığı belirle
                lower_bound = np.array([max(0, c - 20) for c in selected_color[0]])
                upper_bound = np.array([min(255, c + 20) for c in selected_color[0]])
                
                mask = create_color_mask(image, lower_bound, upper_bound)
                contours = extract_contours(mask)
                properties = calculate_contour_properties(contours, mask)
                save_properties_to_excel(properties, 'contour_properties.xlsx')
                print("Özellikler 'contour_properties.xlsx' dosyasına kaydedildi.")

        # Kullanımı
        select_color_and_process('say.jpg')  # Görüntünün yolu

       

    def zoom_in(self):
        if hasattr(self, 'image_label') and self.image is not None:
            new_width = int(self.image.width * 1.25)
            new_height = int(self.image.height * 1.25)
            self.image = self.image.resize((new_width, new_height), Image.BILINEAR)
            self.update_image_display()
        else:
            messagebox.showerror("Hata", "Lütfen önce bir görüntü yükleyin.") 

    def zoom_out(self):
        if hasattr(self, 'image_label') and self.image is not None:
            new_width = int(self.image.width * 0.75)
            new_height = int(self.image.height * 0.75)
            self.image = self.image.resize((new_width, new_height), Image.BILINEAR)
            self.update_image_display()
        else:
            messagebox.showerror("Hata", "Lütfen önce bir görüntü yükleyin.")
            
                    
    def buyut(self):
        scale_factor = simpledialog.askfloat("Büyütme Faktörü", "Büyütme faktörünü girin (örn: 1.5):", minvalue=0.1, maxvalue=10.0)
        if scale_factor:  # None değilse, yani kullanıcı bir değer girdiyse
            self.zoom_image(scale_factor, True)

    def kucult(self):
        scale_factor = simpledialog.askfloat("Küçültme Faktörü", "Küçültme faktörünü girin (örn: 0.75):", minvalue=0.1, maxvalue=1.0)
        if scale_factor:  # None değilse, yani kullanıcı bir değer girdiyse
            self.zoom_image(scale_factor, False)

    def zoom_image(self, scale_factor, zoom_in):
        # Görüntünün yeni boyutlarını hesapla
        new_width = int(self.image.width * scale_factor) if zoom_in else int(self.image.width / scale_factor)
        new_height = int(self.image.height * scale_factor) if zoom_in else int(self.image.height / scale_factor)

        # Görüntüyü yeniden boyutlandır
        self.image = self.image.resize((new_width, new_height), Image.BILINEAR)
        self.update_image_display()
        
    def rotate_image(self):
      if not hasattr(self, 'image') or self.image is None:
          messagebox.showerror("Hata", "Lütfen önce bir görüntü yükleyin.")
          return

    # Kullanıcıdan döndürme açısını iste
      angle = simpledialog.askfloat("Görüntü Döndürme", "Lütfen döndürme açısını girin (derece):", minvalue=-360, maxvalue=360)
    
      if angle is not None:  # Kullanıcı iptal etmediyse
        # Görüntüyü döndür
         self.image = self.image.rotate(angle, expand=1)  # Döndürülmüş görüntüyü self.image olarak güncelle
         self.update_image_display()  # Güncellenmiş görüntüyü göstermek için çağır


    def update_image_display(self):
        photo = ImageTk.PhotoImage(self.image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
        


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
