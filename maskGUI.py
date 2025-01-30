import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from transformers import pipeline
from ultralytics import SAM
import torch

class CombinedSegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Background Removal + SAM Segmentation")
        
        # Initialize save directory
        self.save_dir = "segments"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize models
        try:
            print("Loading background removal model...")
            self.bg_pipe = pipeline("image-segmentation", 
                                  model="briaai/RMBG-1.4", 
                                  trust_remote_code=True)
            print("Loading SAM model...")
            self.sam_model = SAM("sam2_b.pt")
            print("Models loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
            root.destroy()
            return

        # Initialize variables
        self.current_image = None
        self.no_bg_image = None
        self.processed_image = None
        self.scale = 0.8
        
        # Create GUI
        self.setup_gui()
        
    def setup_gui(self):
        # Control buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Load Image", 
                  command=self.load_and_process_image).pack(side=tk.LEFT, padx=5)
        
        # Create frame for image previews
        preview_frame = ttk.Frame(self.root)
        preview_frame.pack(pady=10, expand=True, fill=tk.BOTH)
        
        # Create three columns for previews
        self.preview_frames = []
        self.preview_labels = []
        self.image_labels = []
        
        titles = ["Original Image", "Background Removed", "Segmented Result"]
        
        for i, title in enumerate(titles):
            frame = ttk.Frame(preview_frame)
            frame.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
            preview_frame.grid_columnconfigure(i, weight=1)
            
            # Title label
            title_label = ttk.Label(frame, text=title)
            title_label.pack(pady=5)
            
            # Image label
            image_label = ttk.Label(frame)
            image_label.pack(expand=True)
            
            self.preview_frames.append(frame)
            self.preview_labels.append(title_label)
            self.image_labels.append(image_label)
        
        # Status label
        self.status_label = ttk.Label(self.root, text="Ready")
        self.status_label.pack(pady=5)
        
    def update_preview(self, image, index):
        """Update preview image at specified index"""
        if isinstance(image, np.ndarray):
            # Convert OpenCV image to PIL
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            # If it's neither numpy array nor PIL image, convert to PIL
            image = Image.fromarray(np.array(image))
            
        # Resize for preview
        display_size = (300, 300)  # Fixed size for preview
        image.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage and update label
        photo = ImageTk.PhotoImage(image)
        self.image_labels[index].config(image=photo)
        self.image_labels[index].image = photo  # Keep reference
        
    def load_and_process_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        
        if not file_path:
            return
            
        try:
            # Load and display original image
            self.status_label.config(text="Loading image...")
            self.root.update()
            
            original_image = Image.open(file_path)
            self.update_preview(original_image, 0)
            
            # Step 1: Remove background
            self.status_label.config(text="Removing background...")
            self.root.update()
            
            # Process with background removal
            self.no_bg_image = self.bg_pipe(file_path)
            self.update_preview(self.no_bg_image, 1)
            
            # Convert PIL image to CV2 format for SAM
            cv2_image = cv2.cvtColor(np.array(self.no_bg_image), cv2.COLOR_RGB2BGR)
            
            # Step 2: SAM Processing
            self.status_label.config(text="Performing SAM segmentation...")
            self.root.update()
            
            # Process with SAM (automatic mode, no points needed)
            results = self.sam_model(cv2_image)
            
            if results[0].masks is None:
                messagebox.showinfo("Info", "No segments detected")
                return
                
            # Save original no-background image
            no_bg_path = os.path.join(self.save_dir, 'no_background.png')
            self.no_bg_image.save(no_bg_path)
            
            # Process and save each segment with coordinates
            result_img = cv2_image.copy()
            for i, mask_tensor in enumerate(results[0].masks.data):
                mask = mask_tensor.cpu().numpy()
                if mask.shape != cv2_image.shape[:2]:
                    mask = cv2.resize(mask, (cv2_image.shape[1], cv2_image.shape[0]))
                
                # Get segment coordinates
                coords = np.where(mask > 0.5)
                if len(coords[0]) == 0:
                    continue
                    
                # Calculate bounding box
                min_y, max_y = np.min(coords[0]), np.max(coords[0])
                min_x, max_x = np.min(coords[1]), np.max(coords[1])
                
                # Create and save segment
                segment = np.zeros_like(cv2_image)
                segment[mask > 0.5] = cv2_image[mask > 0.5]
                
                # Save segment image
                segment_path = os.path.join(self.save_dir, f'segment_{i}.png')
                cv2.imwrite(segment_path, segment)
                
                # Save coordinates
                coords_path = os.path.join(self.save_dir, f'segment_{i}_coords.txt')
                with open(coords_path, 'w') as f:
                    f.write(f"Top-left: ({min_x}, {min_y})\n")
                    f.write(f"Bottom-right: ({max_x}, {max_y})\n")
                    f.write(f"Center: ({(min_x + max_x)//2}, {(min_y + max_y)//2})\n")
                
                # Color the segment in result image
                color = np.random.randint(0, 255, 3).tolist()
                result_img[mask > 0.5] = color
            
            # Update final preview
            self.update_preview(result_img, 2)
            
            self.status_label.config(text=f"Processing complete! Results saved in '{self.save_dir}'")
            messagebox.showinfo("Success", 
                              f"Processing complete!\nCheck the '{self.save_dir}' directory for results")
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_label.config(text="Processing failed")

if __name__ == "__main__":
    root = tk.Tk()
    app = CombinedSegmentationGUI(root)
    root.mainloop()