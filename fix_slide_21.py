import fitz
from PIL import Image

pdf_path = "Final Presentation.pptx.pdf"
output_path = "slides_cropped/slide_21.png"

doc = fitz.open(pdf_path)
page = doc.load_page(20) # 0-indexed, so 20 is page 21
pix = page.get_pixmap(dpi=300)
img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

# We do NOT crop this slide to preserve the QR code on the top left
img.save(output_path)
print("Re-extracted slide 21 without crop to preserve QR code.")
