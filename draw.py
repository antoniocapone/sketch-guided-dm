import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
from tkinter import filedialog

class SketchApp:
    def __init__(self, root, width=512, height=512, brush_size=8):
        self.width = width
        self.height = height
        self.brush_size = brush_size

        self.root = root
        self.root.title("Binary Sketch Drawer")

        # Image where we draw
        self.image = Image.new("L", (self.width, self.height), 0)
        self.draw = ImageDraw.Draw(self.image)

        # Canvas for display
        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg="black")
        self.canvas.pack()

        # Convert PIL image to Tkinter-compatible image
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas_img = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Clear", command=self.clear).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Save", command=self.save).pack(side=tk.LEFT)

    def paint(self, event):
        x, y = event.x, event.y
        bbox = [x - self.brush_size // 2, y - self.brush_size // 2,
                x + self.brush_size // 2, y + self.brush_size // 2]
        self.draw.ellipse(bbox, fill=255)
        self.update_canvas()

    def clear(self):
        self.draw.rectangle([0, 0, self.width, self.height], fill=0)
        self.update_canvas()

    def update_canvas(self):
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.canvas_img, image=self.tk_image)

    def save(self):
        filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png")])
        if filename:
            self.image.save(filename)
            print(f"Saved to {filename}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SketchApp(root)
    root.mainloop()
