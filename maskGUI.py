import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from ultralytics import SAM
import torch
from PIL import Image, ImageTk
import os

class SAMDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM Detector GUI")
        
        # Initialize variables
        self.image = None
        self.photo = None
        self.display_image = None
        self.scale = 0.8
        self.points = []
        self.current_mode = "point"  # or "box"
        self.box_start = None
        self.save_dir = "segments"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Load SAM model
        print("Loading SAM model...")
        self.model = SAM("sam2_b.pt")
        print("Model loaded successfully!")
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Left side buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.LEFT)
        
        ttk.Button(btn_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear", command=self.clear_points).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Process", command=self.process_current_image).pack(side=tk.LEFT, padx=5)
        
        # Mode selection
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="point")
        ttk.Radiobutton(mode_frame, text="Point", variable=self.mode_var, 
                       value="point", command=self.change_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Box", variable=self.mode_var,
                       value="box", command=self.change_mode).pack(side=tk.LEFT)
        
        # Point type selection (positive/negative)
        self.point_type_var = tk.StringVar(value="positive")
        point_frame = ttk.Frame(control_frame)
        point_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(point_frame, text="Point Type:").pack(side=tk.LEFT)
        ttk.Radiobutton(point_frame, text="Positive", variable=self.point_type_var,
                       value="positive").pack(side=tk.LEFT)
        ttk.Radiobutton(point_frame, text="Negative", variable=self.point_type_var,
                       value="negative").pack(side=tk.LEFT)
        
        # Canvas for image display
        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
    def change_mode(self):
        self.current_mode = self.mode_var.get()
        self.clear_points()
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is None:
                messagebox.showerror("Error", "Failed to load image")
                return
            self.display_image = self.image.copy()
            self.update_canvas()
            self.clear_points()
            
    def update_canvas(self):
        if self.display_image is not None:
            height, width = self.display_image.shape[:2]
            new_width = int(width * self.scale)
            new_height = int(height * self.scale)
            resized = cv2.resize(self.display_image, (new_width, new_height))
            
            image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            self.photo = ImageTk.PhotoImage(image=image)
            
            self.canvas.config(width=new_width, height=new_height)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
    def on_click(self, event):
        if self.image is None:
            return
            
        if self.current_mode == "point":
            # Add point
            x, y = event.x / self.scale, event.y / self.scale
            self.points.append((int(x), int(y)))
            
            # Draw point
            color = "green" if self.point_type_var.get() == "positive" else "red"
            self.canvas.create_oval(
                event.x-5, event.y-5, event.x+5, event.y+5,
                fill=color, outline=color
            )
        else:  # box mode
            self.box_start = (event.x, event.y)
            
    def on_drag(self, event):
        if self.current_mode == "box" and self.box_start:
            # Refresh display
            self.display_image = self.image.copy()
            self.update_canvas()
            
            # Draw current box
            self.canvas.create_rectangle(
                self.box_start[0], self.box_start[1],
                event.x, event.y,
                outline="green", width=2
            )
            
    def on_release(self, event):
        if self.current_mode == "box" and self.box_start:
            # Calculate box coordinates in original image scale
            x1, y1 = int(self.box_start[0] / self.scale), int(self.box_start[1] / self.scale)
            x2, y2 = int(event.x / self.scale), int(event.y / self.scale)
            self.points = [x1, y1, x2, y2]
            self.box_start = None
            
    def clear_points(self):
        self.points = []
        self.box_start = None
        if self.image is not None:
            self.display_image = self.image.copy()
            self.update_canvas()
            
    def process_current_image(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
            
        if not self.points:
            messagebox.showwarning("Warning", "Please add points or box first")
            return
            
        try:
            if self.current_mode == "point":
                # Prepare points and labels
                points = np.array(self.points)
                labels = np.array([1 if self.point_type_var.get() == "positive" else 0 
                                 for _ in range(len(self.points))])
                results = self.model(self.image, points=points, labels=labels)
            else:  # box mode
                results = self.model(self.image, bboxes=[self.points])
                
            if results[0].masks is None:
                messagebox.showinfo("Info", "No segments detected")
                return
                
            # Visualize results
            self.display_image = self.image.copy()
            
            for i, mask_tensor in enumerate(results[0].masks.data):
                mask = mask_tensor.cpu().numpy()
                if mask.shape != self.image.shape[:2]:
                    mask = cv2.resize(mask, (self.image.shape[1], self.image.shape[0]))
                    
                # Create colored overlay
                color = np.random.randint(0, 255, 3).tolist()
                overlay = np.zeros_like(self.image)
                overlay[mask > 0.5] = color
                
                # Blend with image
                self.display_image = cv2.addWeighted(
                    self.display_image, 1, overlay, 0.5, 0
                )
                
                # Save segment
                segment = np.zeros_like(self.image)
                segment[mask > 0.5] = self.image[mask > 0.5]
                cv2.imwrite(os.path.join(self.save_dir, f'segment_{i}.png'), segment)
                
            self.update_canvas()
            messagebox.showinfo("Success", 
                              f"Processing complete! Segments saved in '{self.save_dir}' directory")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {str(e)}")

def main():
    root = tk.Tk()
    app = SAMDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()