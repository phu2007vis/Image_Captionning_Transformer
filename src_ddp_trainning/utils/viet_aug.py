from PIL import Image
import numpy as np

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.augmentations.geometric.transforms import *
import random
import cv2
import random
import numpy as np
from torchvision import transforms

# class RandomDottedLine(ImageOnlyTransform):
# 	def __init__(self, num_lines=1, p=0.5):
# 		super(RandomDottedLine, self).__init__(p=p)
# 		self.num_lines = num_lines

# 	def apply(self, img, **params):
# 		h, w = img.shape[:2]
# 		for _ in range(self.num_lines):
# 			# Random start and end points
# 			x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
# 			x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
# 			# Random color
# 			# color = tuple(np.random.randint(0, 256, size=3).tolist())
# 			color = random.choice([tuple(np.random.randint(0, 10, size=3).tolist()),tuple(np.random.randint(250, 256, size=3).tolist())])
# 			# Random thickness
# 			thickness = np.random.randint(5, 10)
# 			# Draw dotted or dashed line
# 			line_type = random.choice(["dotted", "dashed", "solid"])
# 			if line_type != "solid":
# 				self._draw_dotted_line(
# 					img, (x1, y1), (x2, y2), color, thickness, line_type
# 				)
# 			else:
# 				cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# 		return img

	# def _draw_dotted_line(self, img, pt1, pt2, color, thickness, line_type):
	# 	# Calculate the distance between the points
	# 	dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
	# 	# Number of segments
	# 	num_segments = max(int(dist // 5), 1)
	# 	# Generate points along the line
	# 	x_points = np.linspace(pt1[0], pt2[0], num_segments)
	# 	y_points = np.linspace(pt1[1], pt2[1], num_segments)
	# 	# Draw segments
	# 	for i in range(num_segments - 1):
	# 		if line_type == "dotted" and i % 2 == 0:
	# 			pt_start = (int(x_points[i]), int(y_points[i]))
	# 			pt_end = (int(x_points[i]), int(y_points[i]))
	# 			cv2.circle(img, pt_start, thickness, color, -1)
	# 		elif line_type == "dashed" and i % 4 < 2:
	# 			pt_start = (int(x_points[i]), int(y_points[i]))
	# 			pt_end = (int(x_points[i + 1]), int(y_points[i + 1]))
	# 			cv2.line(img, pt_start, pt_end, color, thickness)
	# 	return img

	# def get_transform_init_args_names(self):
	# 	return ("num_lines",)
class RandomDottedLine(ImageOnlyTransform):
	def __init__(self, num_lines=1, p=1):
		super(RandomDottedLine, self).__init__(p=p)
		self.num_lines = num_lines

	def apply(self, img, **params):
		self.img = img
		if random.random() < 0.4:
			self.random_setup_horizal()
			self.img = self.main_draw(self.img)
		if random.random() < 0.2:
			self.random_left_vertical()
			self.img = self.main_draw(self.img)
		if random.random() < 0.2:
			self.random_right_vertical()
			self.img = self.main_draw(self.img)
		return self.img
	def main_draw(self,img):
		
		if self.line_type != "solid":
			self._draw_dotted_line(
				img, (self.x1, self.y1), (self.x2, self.y2), self.color, self.thickness, self.line_type
			)
		else:
			cv2.line(img, (self.x1, self.y1), (self.x2, self.y2), self.color, self.thickness)
		return img
	def random_generally(self):
		self.color = tuple(np.random.randint(0, 20, size=3).tolist())
		self.thickness = np.random.randint(1,4 )
		self.line_type = random.choice(["solid"])
	def random_setup_horizal(self):
		h, w = self.img.shape[:2]
		max_height = int(0.6*h)
		max_left = int(0.2*w)
		max_right = int(0.8*w)
		self.x1, self.y1 = np.random.randint(0, max_left), np.random.randint(max_height, h)
		self.x2, self.y2 = np.random.randint(max_right, w), np.random.randint(max_height, h)
		self.random_generally()
	def random_left_vertical(self):
	 
		h, w = self.img.shape[:2]
		max_width = int(0.15*w)
		max_top = int(0.1*h)
		max_bottom = int(0.9*h)
  
		self.x1, self.y1 = np.random.randint(0, max_width), np.random.randint(0,max_top)
		self.x2, self.y2 = np.random.randint(0,max_width), np.random.randint(max_bottom, h)
		self.random_generally()
	def random_right_vertical(self):
	 
		h, w = self.img.shape[:2]
		max_width = int(0.85*w)
		max_top = int(0.1*h)
		max_bottom = int(0.9*h)
  
		self.x1, self.y1 = np.random.randint(max_width, w), np.random.randint(0,max_top)
		self.x2, self.y2 = np.random.randint(max_width,w), np.random.randint(max_bottom, h)
		self.random_generally()
  
	def _draw_dotted_line(self, img, pt1, pt2, color, thickness, line_type):
		# Calculate the distance between the points
		dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
		# Number of segments
		num_segments = max(int(dist // 5), 1)
		# Generate points along the line
		x_points = np.linspace(pt1[0], pt2[0], num_segments)
		y_points = np.linspace(pt1[1], pt2[1], num_segments)
		# Draw segments
		for i in range(num_segments - 1):
			if line_type == "dotted" and i % 2 == 0:
				pt_start = (int(x_points[i]), int(y_points[i]))
				pt_end = (int(x_points[i]), int(y_points[i]))
				cv2.circle(img, pt_start, thickness, color, -1)
			elif line_type == "dashed" and i % 4 < 2:
				pt_start = (int(x_points[i]), int(y_points[i]))
				pt_end = (int(x_points[i + 1]), int(y_points[i + 1]))
				cv2.line(img, pt_start, pt_end, color, thickness)
		return img

	def get_transform_init_args_names(self):
		return ("num_lines",)
	

class ImgAugTransformV2:
	def __init__(self):
		self.aug = A.Compose(
				[
		
				# A.AdditiveNoise (noise_type='gaussian', spatial_mode='per_pixel', p= 0.8,approximation = 0.1),
				# Perspective(scale=(0.02, 0.08), p=0.3,fit_output = True,fill = (255,255,255)),
			
				# A.RandomRain(brightness_coefficient=1.0, drop_length=2, drop_width=1, drop_color = (255, 255, 255), blur_value=5, rain_type = 'drizzle', p=0.3), 
				# A.ColorJitter(brightness = (0.3,1.2),contrast = [0.1,1.8],p = 0.8, saturation = 0.4 ,hue = 0.2),
				# A.Defocus(radius=(1,3),alias_blur = (0.1,0.5),p = 0.7),
				# A.Downscale(scale_range = (0.04,0.09), interpolation_pair  = {'upscale':  1,'downscale': 3},p = 1),
				# A.MotionBlur(blur_limit=16, p= 0.8),
				
			
			
				# RandomDottedLine(3),
				# A.Rotate(limit=30, p=0.7),
				# A.RandomRain(brightness_coefficient=1.0, drop_length=2, drop_width=1, drop_color = (255, 255, 255), blur_value=5, rain_type = 'drizzle', p=0.3), 
				RandomDottedLine(2),
				# A.ColorJitter(brightness = 0.2,contrast = 0.2,p = 0.8),
				
				A.OneOf([
					#add black pixels noise
					A.OneOf([
							A.RandomRain(brightness_coefficient=1.0, drop_length=2, drop_width=2, drop_color = (0, 0, 0), blur_value=1, rain_type = 'drizzle', p=0.1), 
							A.RandomShadow(p=1),
							A.PixelDropout(p=1),
						], p=0.33),

					#add white pixels noise
					A.OneOf([
							A.PixelDropout(dropout_prob=0.5,drop_value=255,p=1),
							A.RandomRain(brightness_coefficient=1.0, drop_length=2, drop_width=2, drop_color = (255, 255, 255), blur_value=1, rain_type =  'drizzle', p=1), 
						], p=0.7),
					], p=1),
					
				A.OneOf([
						A.ShiftScaleRotate(shift_limit=0, scale_limit=0.25, rotate_limit=2, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255),p=1),
						A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=8, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255),p=1),
						A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.15, rotate_limit=11, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255),p=1),  
						A.Affine(shear=random.randint(-5, 5),mode=cv2.BORDER_CONSTANT, cval=(255,255,255), p=1)          
				], p=0.5),
				
				A.Blur(blur_limit=3,p=0.25),

			]
		)
		# self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
		self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	
	def __call__(self, img):
		img = np.copy(np.asarray(img))
		kernel  = self.kernel
	
		# if random.randint(1, 3) == 1:
			# erosion because the image is not inverted
			# img = cv2.dilate(img, kernel,iterations=random.randint(1, 3))
		# if random.randint(1, 5) == 1:
		# 	# dilation because the image is not inverted
		if random.randint(1, 2) == 1:
			img = cv2.erode(img, kernel, iterations=random.randint(2, 4))
		# if random.randint(1, 6) == 1:
		# 	# erosion because the image is not inverted
		# 	img = cv2.dilate(img, kernel,iterations=random.randint(1, 1))
		# img = cv2.dilate(img, kernel,iterations=random.randint(3, 3))
		transformed = self.aug(image=img)
		img = transformed["image"]
		img = Image.fromarray(img)
		return img
	

import torchvision.transforms as transforms
from PIL import Image
import random


# Define a custom transformation to restrict shift **only upwards**
class UpwardsShift(transforms.RandomAffine):
	def __init__(self, max_shift=0.2,fill = (255,255,255)):
		super().__init__(degrees=0, translate=(0, max_shift),fill=fill)  # Max 30% shift in Y

	def get_params(self, degrees, translate, scale_ranges, shears, img_size):
		"""Override get_params to ensure only upward translation"""
		angle = 0  # No rotation
		img_width, img_height = img_size

		# Extract allowed max shift percentage
		max_shift_x, max_shift_y = translate  # Should be single values, not tuples!

		# Random translation values
		tx = random.uniform(-max_shift_x, max_shift_x) * img_width  # Horizontal shift
		ty = -random.uniform(-0.22, max_shift_y) * img_height  # Only shift **upwards**

		scale = 1.0  # Fixed scale (was mistakenly a tuple before)
		shear = 0.0

		return angle, (tx, ty), scale, shear

import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import random

class Stretch:
	def __init__(self, p, rng=None):
		self.rng = np.random.default_rng() if rng is None else rng
		self.tps = cv2.createThinPlateSplineShapeTransformer()
		self.prob = p

	def __call__(self, img):
		if self.rng.uniform(0, 1) > self.prob:
			return img

		w, h = img.size
		img = np.asarray(img)
		srcpt = []
		dstpt = []

		w_33 = 0.33 * w
		w_50 = 0.50 * w
		w_66 = 0.66 * w

		h_50 = 0.50 * h

		p = 0

		b = [.2, .3, .4]
		frac = random.choice(b)  # Randomly select frac like Curve uses random.choice for rmin

		# left-most
		srcpt.append([p, p])
		srcpt.append([p, h - p])
		srcpt.append([p, h_50])
		x = self.rng.uniform(0, frac) * w_33
		dstpt.append([p + x, p])
		dstpt.append([p + x, h - p])
		dstpt.append([p + x, h_50])

		# 2nd left-most
		srcpt.append([p + w_33, p])
		srcpt.append([p + w_33, h - p])
		x = self.rng.uniform(-frac, frac) * w_33
		dstpt.append([p + w_33 + x, p])
		dstpt.append([p + w_33 + x, h - p])

		# 3rd left-most
		srcpt.append([p + w_66, p])
		srcpt.append([p + w_66, h - p])
		x = self.rng.uniform(-frac, frac) * w_33
		dstpt.append([p + w_66 + x, p])
		dstpt.append([p + w_66 + x, h - p])

		# right-most
		srcpt.append([w - p, p])
		srcpt.append([w - p, h - p])
		srcpt.append([w - p, h_50])
		x = self.rng.uniform(-frac, 0) * w_33
		dstpt.append([w - p + x, p])
		dstpt.append([w - p + x, h - p])
		dstpt.append([w - p + x, h_50])

		n = len(dstpt)
		matches = [cv2.DMatch(i, i, 0) for i in range(n)]
		dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
		src_shape = np.asarray(srcpt).reshape((-1, n, 2))
		self.tps.estimateTransformation(dst_shape, src_shape, matches)
		img = self.tps.warpImage(img, borderValue=(255, 255, 255))
		img = Image.fromarray(img)

		return img


class Distort:
	def __init__(self, p, rng=None):
		self.rng = np.random.default_rng() if rng is None else rng
		self.tps = cv2.createThinPlateSplineShapeTransformer()
		self.prob = p

	def __call__(self, img):
		if self.rng.uniform(0, 1) > self.prob:
			return img

		w, h = img.size
		img = np.asarray(img)
		srcpt = []
		dstpt = []

		w_33 = 0.33 * w
		w_50 = 0.50 * w
		w_66 = 0.66 * w

		h_50 = 0.50 * h

		p = 0

		b = [.2, .3, .4]
		frac = random.choice(b)  # Randomly select frac like Curve uses random.choice for rmin

		# top pts
		srcpt.append([p, p])
		x = self.rng.uniform(0, frac) * w_33
		y = self.rng.uniform(0, frac) * h_50
		dstpt.append([p + x, p + y])

		srcpt.append([p + w_33, p])
		x = self.rng.uniform(-frac, frac) * w_33
		y = self.rng.uniform(0, frac) * h_50
		dstpt.append([p + w_33 + x, p + y])

		srcpt.append([p + w_66, p])
		x = self.rng.uniform(-frac, frac) * w_33
		y = self.rng.uniform(0, frac) * h_50
		dstpt.append([p + w_66 + x, p + y])

		srcpt.append([w - p, p])
		x = self.rng.uniform(-frac, 0) * w_33
		y = self.rng.uniform(0, frac) * h_50
		dstpt.append([w - p + x, p + y])

		# bottom pts
		srcpt.append([p, h - p])
		x = self.rng.uniform(0, frac) * w_33
		y = self.rng.uniform(-frac, 0) * h_50
		dstpt.append([p + x, h - p + y])

		srcpt.append([p + w_33, h - p])
		x = self.rng.uniform(-frac, frac) * w_33
		y = self.rng.uniform(-frac, 0) * h_50
		dstpt.append([p + w_33 + x, h - p + y])

		srcpt.append([p + w_66, h - p])
		x = self.rng.uniform(-frac, frac) * w_33
		y = self.rng.uniform(-frac, 0) * h_50
		dstpt.append([p + w_66 + x, h - p + y])

		srcpt.append([w - p, h - p])
		x = self.rng.uniform(-frac, 0) * w_33
		y = self.rng.uniform(-frac, 0) * h_50
		dstpt.append([w - p + x, h - p + y])

		n = len(dstpt)
		matches = [cv2.DMatch(i, i, 0) for i in range(n)]
		dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
		src_shape = np.asarray(srcpt).reshape((-1, n, 2))
		self.tps.estimateTransformation(dst_shape, src_shape, matches)
		img = self.tps.warpImage(img, borderValue=(255, 255, 255))
		img = Image.fromarray(img)

		return img


class Curve:
	def __init__(self, p, square_side=224, rng=None):
		self.tps = cv2.createThinPlateSplineShapeTransformer()
		self.side = square_side
		self.rng = np.random.default_rng() if rng is None else rng
		self.prob = p

	def __call__(self, img):
		if self.rng.uniform(0, 1) > self.prob:
			return img

		orig_w, orig_h = img.size

		if orig_h != self.side or orig_w != self.side:
			img = img.resize((self.side, self.side), Image.BICUBIC)

		isflip = self.rng.uniform(0, 1) > 0.5
		if isflip:
			img = ImageOps.flip(img)

		img = np.asarray(img)
		w = self.side
		h = self.side
		w_25 = 0.25 * w
		w_50 = 0.50 * w
		w_75 = 0.75 * w

		b = [1.1, .95, .8]
		rmin = random.choice(b)

		r = self.rng.uniform(rmin, rmin + .1) * h
		x1 = (r ** 2 - w_50 ** 2) ** 0.5
		h1 = r - x1

		t = self.rng.uniform(0.4, 0.5) * h

		w2 = w_50 * t / r
		hi = x1 * t / r
		h2 = h1 + hi

		sinb_2 = ((1 - x1 / r) / 2) ** 0.5
		cosb_2 = ((1 + x1 / r) / 2) ** 0.5
		w3 = w_50 - r * sinb_2
		h3 = r - r * cosb_2

		w4 = w_50 - (r - t) * sinb_2
		h4 = r - (r - t) * cosb_2

		w5 = 0.5 * w2
		h5 = h1 + 0.5 * hi
		h_50 = 0.50 * h

		srcpt = [(0, 0), (w, 0), (w_50, 0), (0, h), (w, h), (w_25, 0), (w_75, 0), (w_50, h), (w_25, h), (w_75, h),
				 (0, h_50), (w, h_50)]
		dstpt = [(0, h1), (w, h1), (w_50, 0), (w2, h2), (w - w2, h2), (w3, h3), (w - w3, h3), (w_50, t), (w4, h4),
				 (w - w4, h4), (w5, h5), (w - w5, h5)]

		n = len(dstpt)
		matches = [cv2.DMatch(i, i, 0) for i in range(n)]
		dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
		src_shape = np.asarray(srcpt).reshape((-1, n, 2))
		self.tps.estimateTransformation(dst_shape, src_shape, matches)
		img = self.tps.warpImage(img, borderValue=(255, 255, 255))
		img = Image.fromarray(img)

		if isflip:
			img = ImageOps.flip(img)
			rect = (0, self.side // 2, self.side, self.side)
		else:
			rect = (0, 0, self.side, self.side // 2)

		img = img.crop(rect)
		img = img.resize((orig_w, orig_h), Image.BICUBIC)
		return img

class WrapAug:
	def __init__(self, p):
		self.p_distort = p[0]
		self.p_strech = p[1]
		self.p_cur = p[2]
		self.augs = [ Curve(p=self.p_cur),Stretch(p=self.p_strech), Distort(p=self.p_distort)]
	
	def __call__(self, img):
		for aug in self.augs:
			img = aug(img)
		return img



if __name__ == "__main__":
	image_path  = "/work/21013187/phuoc/Image_Captionning_Transformer/data/OCR_Phuoc/train/images/1PlateBaza493.jpg"
	out_path = "/work/21013187/phuoc/Image_Captionning_Transformer/data/test/test.png"
	transform = ImgAugTransformV2()
	pil_image = Image.open(image_path)
	pil_image = transform(pil_image)
	pil_image.save(out_path)
	import os
	import sys
	SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
	sys.path.append(os.path.dirname(SCRIPT_DIR))
	image_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/OCR_Phuoc/train/images"
	image_save_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/test"
	image_height =  224
	image_min_width =  120
	image_max_width=  448
	from utils.resize_image import process_image
  
	os.makedirs(image_folder,exist_ok= True)
	for image_name in os.listdir(image_folder):
		
		image_path = os.path.join(image_folder,image_name)
		save_path  = os.path.join(image_save_folder,image_name)
		pil_image = Image.open(image_path)
		pil_image = process_image(pil_image,image_height,image_min_width,image_max_width)
		pil_image = transform(pil_image)
		pil_image.save(save_path)
	
	
	