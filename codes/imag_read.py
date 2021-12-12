# helper to load data from PNG image files
import imageio
# glob helps select multiple files using patterns
import glob
import numpy
# library for plotting arrays
import matplotlib.pyplot as plt

image_file_name = 'images/3.1.PNG'
# load image data from png files into an array
img_array = imageio.imread(image_file_name, as_gray=True)
# reshape from 28x28 to list of 784 values, invert values
img_data  = 255.0 - img_array.reshape(784)
# then scale data to range from 0.01 to 1.0
img_data = (img_data / 255.0 * 0.99) + 0.01
# append label and image data  to test data set
# print(img_data)
plt.imshow(img_data.reshape(28,28), cmap='Greys', interpolation='None') 
plt.show()