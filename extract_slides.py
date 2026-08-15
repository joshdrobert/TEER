import fitz  # PyMuPDF
from PIL import Image
import os

pdf_path = "Final Presentation.pptx.pdf"
output_dir = "slides_cropped"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

doc = fitz.open(pdf_path)

# Crop ratio (e.g., top 15% cropped out)
crop_top_ratio = 0.15

for i in range(len(doc)):
    page = doc.load_page(i)
    pix = page.get_pixmap(dpi=300)
    
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Crop top 15%
    width, height = img.size
    top = int(height * crop_top_ratio)
    cropped_img = img.crop((0, top, width, height))
    
    # We can also resize back to 1080p width to keep aspect ratio if needed, or just save
    cropped_img.save(f"{output_dir}/slide_{i+1:02d}.png")
    
print(f"Extracted and cropped {len(doc)} slides.")
