import numpy as np
import os
import sys
import argparse
import cv2
import skimage

def average_color(block):
    average_row_color = np.average(block, axis=0)
    average_color = np.average(average_row_color, axis=0)
    return average_color

def dominant_color(block):
	pixels = np.float32(block.reshape(-1, 3))
	n_colors = 5
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
	flags = cv2.KMEANS_RANDOM_CENTERS
	_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
	_, counts = np.unique(labels, return_counts=True)
	return palette[np.argmax(counts)]

def generate_mosaic(reshaped_img, block_size = 5):
	output = np.zeros(reshaped_img.shape)
	height, width = reshaped_img.shape[:2]

	rows, cols = int(height / block_size), int(width / block_size)

	for r in range(rows):
		for c in range(cols): 
			h_pos = (block_size * r)
			w_pos = (block_size * c)
			block = reshaped_img[h_pos:h_pos+block_size, w_pos:w_pos+block_size]
			color = dominant_color(block)
			output[h_pos:h_pos+block_size, w_pos:w_pos+block_size] = color
			

	cv2.imwrite('mosaic.png', output)
	return output

def reshape_img(img, block_size = 5):
	height, width = img.shape[:2]
	w_crop = int((width % block_size))
	h_crop = int((height % block_size))
	if w_crop or h_crop:
		w_adjust = 1 if (w_crop % 2) else 0
		h_adjust = 1 if (h_crop % 2) else 0
		min_h, min_w = h_crop // 2, w_crop // 2
		img = img[min_h:height-(min_h + h_adjust), min_w: width-(min_w + w_adjust)]
	return img

def main():
	parser = argparse.ArgumentParser(description = "Generate a mosaic.")
	parser.add_argument('img')
	parser.add_argument('block_size')
	args = parser.parse_args()

	img = args.img
	block_size = int(args.block_size)

	# read in image
	img = cv2.imread(img)
	reshaped_img = reshape_img(img, block_size)

	mosaic_img = generate_mosaic(reshaped_img, block_size)


if __name__ == "__main__":
	try:
		main()
	except (KeyboardInterrupt, SystemExit):
		sys.exit()
