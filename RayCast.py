from PIL import Image, ImageDraw
import numpy as np
from numpy.ma.core import floor
from pathlib import Path


def is_white(pixel):
    return pixel[0] > 150 and pixel[1] > 150 and pixel[2] > 150


class RayCast:
    def __init__(self, image_path, fov, ray_nb):
        self.image = Image.open(image_path)
        self.name = Path(image_path)
        width, height = self.image.size
        self.width = width
        self.height = height
        self.fov = fov
        self.ray_nb = ray_nb
        self.pixels = np.array(self.image)
        self.canvas = ImageDraw.Draw(self.image)

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
        half_fov = self.fov / 2
        angle_step = self.fov / (self.ray_nb - 1)
        center_x = self.width // 2
        center_y = self.height - 1
        for i in range(self.ray_nb):
            ray_angle = -half_fov + i * angle_step - 90
            intersection = self.cast_ray(np.deg2rad(ray_angle))
            self.canvas.line([center_x, center_y, intersection[0], intersection[1]], fill="blue", width=1)
        self.image.save("RayCastOutput/" + self.name.stem + self.name.suffix)
