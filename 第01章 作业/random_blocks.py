from PIL import Image, ImageDraw
import random

img_size = 1024
block_size = 8
img = Image.new("RGB", (img_size, img_size), "white")
draw = ImageDraw.Draw(img)
for y in range(0, img_size, block_size):
    for x in range(0, img_size, block_size):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.rectangle([x, y, x + block_size, y + block_size], fill=color)

img.save("random_blocks.png")
