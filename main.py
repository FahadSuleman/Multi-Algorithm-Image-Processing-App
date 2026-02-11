import os
import torch
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torchvision.transforms as T
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def process_and_save_image(image_path, output_folder, algorithms):
    try:
        image = Image.open(image_path)
        image_name = os.path.basename(image_path)
        transform = T.ToTensor()
        tensor_image = transform(image).unsqueeze(0).to(device)
        
        for algorithm in algorithms:
            np_image = tensor_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            np_image = (np_image * 255).astype(np.uint8)  # Ensure values are in range [0, 255]

            if algorithm == "Grayscale":
                processed_tensor = tensor_image.mean(dim=1, keepdim=True)
            elif algorithm == "Blur":
                blurred_image = cv2.GaussianBlur(np_image, (5, 5), 0)
                processed_tensor = torch.tensor(blurred_image.transpose(2, 0, 1)).unsqueeze(0).to(device)
            elif algorithm == "Edge Detection":
                edges = cv2.Canny(np_image, 100, 200)
                processed_tensor = torch.tensor(edges).unsqueeze(0).unsqueeze(0).to(device)
            elif algorithm == "Histogram Equalization":
                yuv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2YUV)
                yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
                equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
                processed_tensor = torch.tensor(equalized_image.transpose(2, 0, 1)).unsqueeze(0).to(device)
            elif algorithm == "Median Filter":
                filtered_image = cv2.medianBlur(np_image, 5)  # Median filter
                processed_tensor = torch.tensor(filtered_image.transpose(2, 0, 1)).unsqueeze(0).to(device)
            elif algorithm == "Bilateral Filter":
                filtered_image = cv2.bilateralFilter(np_image, d=9, sigmaColor=75, sigmaSpace=75)  # Bilateral filter
                processed_tensor = torch.tensor(filtered_image.transpose(2, 0, 1)).unsqueeze(0).to(device)
            elif algorithm == "Median Blur":
                blurred_image = cv2.medianBlur(np_image, 5)  # Median blur
                processed_tensor = torch.tensor(blurred_image.transpose(2, 0, 1)).unsqueeze(0).to(device)
            else:
                raise ValueError("Unknown algorithm selected")

            processed_image = T.ToPILImage()(processed_tensor.squeeze(0).cpu())
            algorithm_folder = os.path.join(output_folder, algorithm)
            if not os.path.exists(algorithm_folder):
                os.makedirs(algorithm_folder)
            output_image_path = os.path.join(algorithm_folder, f"processed_{image_name}")
            processed_image.save(output_image_path)

        return output_image_path
    except Exception as e:
        return str(e)

def process_images():
    input_folder = input_folder_entry.get()
    selected_algorithms = [alg for alg, var in algorithm_vars.items() if var.get()]
    
    if not os.path.exists(input_folder):
        messagebox.showerror("Error", "Input folder does not exist.")
        return
    if not selected_algorithms:
        messagebox.showerror("Error", "Please select at least one algorithm.")
        return
    
    output_folder = os.path.join("Processed_Images")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_paths = [os.path.join(input_folder, img) for img in os.listdir(input_folder) 
                   if img.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]
    total_images = len(image_paths)
    
    if total_images == 0:
        messagebox.showerror("Error", "No valid images found in the input folder.")
        return
    
    progress_bar.set(0)  # Reset the progress bar to empty
    progress_bar.configure(progress_color="#666666")  # Set initial gray color
    loading_label.configure(text="Processing...")

    def process_thread():
        with ThreadPoolExecutor() as executor:
            futures = []
            for idx, img_path in enumerate(image_paths):
                futures.append(executor.submit(process_and_save_image, img_path, output_folder, selected_algorithms))
                
            for idx in range(total_images):
                # Update progress bar after each image is processed
                root.after(100, lambda: progress_bar.set((idx + 1) / total_images))
                
        # Change progress bar color to blue after completion
        root.after(100, lambda: progress_bar.configure(progress_color="blue"))
        loading_label.configure(text="Processing Complete!")
        messagebox.showinfo("Success", "Image processing complete!")

    threading.Thread(target=process_thread).start()

def select_folder():
    folder = filedialog.askdirectory()
    input_folder_entry.delete(0, "end")
    input_folder_entry.insert(0, folder)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Parallel Image Processing")
root.geometry("900x700")
root.configure(bg="#000000")

# Heading with opacity background
heading_frame = ctk.CTkFrame(root, fg_color="#666666", corner_radius=10, width=400, height=30)
heading_frame.place(relx=0.5, rely=0.05, anchor="center")
heading_label = ctk.CTkLabel(heading_frame, text="Parallel Image Processing", font=("Arial", 18, "bold"), text_color="black")
heading_label.place(relx=0.5, rely=0.5, anchor="center")

# Input folder entry
input_folder_frame = ctk.CTkFrame(root, fg_color="#666666", corner_radius=10, width=400, height=80)
input_folder_frame.place(relx=0.5, rely=0.2, anchor="center")
input_folder_label = ctk.CTkLabel(input_folder_frame, text="Input Folder", font=("Arial", 12,"bold"))
input_folder_label.place(relx=0.5, rely=0.1, anchor="center")
input_folder_entry = ctk.CTkEntry(input_folder_frame, width=250)
input_folder_entry.place(relx=0.5, rely=0.45, anchor="center")
ctk.CTkButton(input_folder_frame, text="Browse", command=select_folder).place(relx=0.5, rely=0.8, anchor="center")

# Algorithm selection
algorithm_frame = ctk.CTkFrame(root, fg_color="#666666", corner_radius=10, width=400, height=277)
algorithm_frame.place(relx=0.5, rely=0.58, anchor="center")
algorithm_label = ctk.CTkLabel(algorithm_frame, text="Select Algorithm", font=("Arial", 12,"bold"))
algorithm_label.place(relx=0.5, rely=0.05, anchor="center")

# Create a sub-frame for checkboxes
checkbox_frame = ctk.CTkFrame(algorithm_frame, fg_color="#666666", corner_radius=0,width=350,height=150)
checkbox_frame.place(relx=0.5,rely=0.5 ,anchor="center")

algorithm_vars = {alg: ctk.BooleanVar() for alg in ["Grayscale", "Blur", "Edge Detection", "Histogram Equalization", "Median Filter", "Bilateral Filter", "Median Blur"]}
for alg,var in algorithm_vars.items():
    ctk.CTkCheckBox(checkbox_frame,text=alg ,variable=var).pack(pady=5 ,anchor="w")

# Process button
process_button = ctk.CTkButton(root,text="Process Images" ,command=process_images)
process_button.place(relx=0.5,rely=0.89 ,anchor="center")

# Progress bar and loading label
progress_bar_frame = ctk.CTkFrame(root ,fg_color="#666666" ,corner_radius=10,width=400,height=30)
progress_bar_frame.place(relx=0.5,rely=0.95 ,anchor="center")
progress_bar = ctk.CTkProgressBar(progress_bar_frame ,orientation="horizontal" ,width=200)
progress_bar.place(relx=0.5,rely=0.5 ,anchor="center")
loading_label = ctk.CTkLabel(root,text="" ,font=("Arial" ,14))
loading_label.place(relx=0.5,rely=0.9 ,anchor="center")

root.mainloop()
