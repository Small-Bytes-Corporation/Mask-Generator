from PIL import Image, ImageDraw
import numpy as np
from numpy.ma.core import floor
from pathlib import Path
from math import sqrt

def is_white(pixel):
    return int(pixel[0]) + int(pixel[1]) + int(pixel[2]) > 100

class RayCast:
    def __init__(self, image_path, fov, ray_nb):
        self.image = Image.open(image_path).convert("RGB")
        self.name = Path(image_path)
        width, height = self.image.size
        self.width = width
        self.height = height
        self.fov = fov
        self.ray_nb = ray_nb
        self.pixels = np.array(self.image)
        self.canvas = ImageDraw.Draw(self.image)
        os.makedirs("RayCastOutput", exist_ok=True)

    def cast_ray(self, angle):
        dx = np.cos(angle)
        dy = np.sin(angle)
        x = float(self.width // 2)
        y = self.height - 1
        while 0 <= x < self.width and 0 <= y < self.height:
            int_x, int_y = int(floor(x)), int(floor(y))
            if is_white(self.pixels[int_y, int_x]):
                return int_x, int_y
            x += dx
            y += dy
        return x, y


    def run(self):
        distance_array = []
        half_fov = self.fov / 2
        angle_step = self.fov / (self.ray_nb - 1)
        center_x = self.width // 2
        center_y = self.height - 1
        for i in range(self.ray_nb):
            ray_angle = -half_fov + i * angle_step - 90
            intersection = self.cast_ray(np.deg2rad(ray_angle))
            dist_x = intersection[0] - center_x
            dist_y = intersection[1] - center_y
            distance_array.append(sqrt(dist_x * dist_x + dist_y * dist_y))
            self.canvas.line([center_x, center_y, intersection[0], intersection[1]], fill="blue", width=1)
        self.image.save("RayCastOutput/" + self.name.stem + "-" + str(self.fov) + "-" + str(self.ray_nb) + self.name.suffix)
        return distance_array
