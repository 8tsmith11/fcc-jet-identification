from qiskit_machine_learning.utils import algorithm_globals
import numpy as np

def generate_line_dataset(num_images, width=4, height=2, min_l=2, max_l=-1):
    if max_l == -1:
        max_l = min_l
    images = []
    labels = []

    pixels = width * height

    horizontals = 0 # number of possible horizontal lines
    verticals = 0 # number of possible vertical lines

    # count possible arrangements for each line length
    for length in range(min_l, max_l + 1):
        if length <= width:
            horizontals += (width - length + 1) * height
        if length <= height:
            verticals += (height - length + 1) * width

    hor_array = np.zeros((horizontals, width*height))
    ver_array = np.zeros((verticals, width*height))

    j = 0
    for i in range(0, 7):
        if i != 3:
            hor_array[j][i] = np.pi / 2
            hor_array[j][i + 1] = np.pi / 2
            j += 1

    for l in range(min_l, max_l + 1):
        j = 0
        for i in range(0, pixels):
            col = i % width
            if col + l <= width:
                hor_array[j][i:i+l-1] = np.pi / 2
                j == 1
             

    j = 0
    for i in range(0, 4):
        ver_array[j][i] = np.pi / 2
        ver_array[j][i + 4] = np.pi / 2
        j += 1

    for n in range(num_images):
        rng = algorithm_globals.random.integers(0, 2)
        if rng == 0:
            labels.append(-1)   
            random_image = algorithm_globals.random.integers(0, 6)
            images.append(np.array(hor_array[random_image]))
        elif rng == 1:
            labels.append(1)
            random_image = algorithm_globals.random.integers(0, 4)
            images.append(np.array(ver_array[random_image]))

        # Create noise
        for i in range(8):
            if images[-1][i] == 0:
                images[-1][i] = algorithm_globals.random.uniform(0, np.pi / 4)
    return images, labels