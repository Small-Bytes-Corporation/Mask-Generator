#!/bin/python3

from RayCast import RayCast
from sys import argv
import os

if __name__ == '__main__':
    fov = 90
    ray_nb = 10
    for filename in os.listdir("MaskOutput/"):
        caster = RayCast("MaskOutput/" + filename, fov, ray_nb)
        print(caster.run())
    exit(0)
