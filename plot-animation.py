import numpy as np
import matplotlib.pyplot as plt

from src.util import make_3d_animation
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


radius = 25

fields = np.load(f'./field_arr_discrete.npy', )
animation_filename = f'field_arr_discrete'
make_3d_animation(fields, radius=radius, filename=animation_filename, interpolation="bilinear")