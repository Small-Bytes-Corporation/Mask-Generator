#!/bin/python3

from RayCast import RayCast
from sys import argv
import os

fov = 90
ray_nb = 10
input_dir = "MaskLines/"

if __name__ == '__main__':
    for filename in os.listdir(input_dir):
        caster = RayCast(input_dir + filename, fov, ray_nb)
        print(caster.run())
    exit(0)
