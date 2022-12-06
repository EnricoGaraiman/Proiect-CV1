import numpy as np

rgb2gray = lambda img: np.uint8(0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.144 * img[:, :, 2])
