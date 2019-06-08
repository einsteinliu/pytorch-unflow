#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys

try:
	from correlation import correlation # the custom cost volume layer
except:
	sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python
# end

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:3])) >= 41) # requires at least pytorch version 0.4.1

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'css'
arguments_strFirst = './images/first.png'
arguments_strSecond = './images/second.png'
arguments_strOut = './out.flo'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use
	if strOption == '--first' and strArgument != '': arguments_strFirst = strArgument # path to the first frame
	if strOption == '--second' and strArgument != '': arguments_strSecond = strArgument # path to the second frame
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

Backward_tensorGrid = {}

def Backward(tensorInput, tensorFlow):
	if str(tensorFlow.size()) not in Backward_tensorGrid:
		tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
		tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

		Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
	# end

	tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
# end

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		class Upconv(torch.nn.Module):
			def __init__(self):
				super(Upconv, self).__init__()

				self.moduleSixOut = torch.nn.Conv2d(in_channels=1024, out_channels=2, kernel_size=3, stride=1, padding=1)

				self.moduleSixUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)

				self.moduleFivNext = torch.nn.Sequential(
					torch.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFivOut = torch.nn.Conv2d(in_channels=1026, out_channels=2, kernel_size=3, stride=1, padding=1)

				self.moduleFivUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)

				self.moduleFouNext = torch.nn.Sequential(
					torch.nn.ConvTranspose2d(in_channels=1026, out_channels=256, kernel_size=4, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFouOut = torch.nn.Conv2d(in_channels=770, out_channels=2, kernel_size=3, stride=1, padding=1)

				self.moduleFouUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)

				self.moduleThrNext = torch.nn.Sequential(
					torch.nn.ConvTranspose2d(in_channels=770, out_channels=128, kernel_size=4, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleThrOut = torch.nn.Conv2d(in_channels=386, out_channels=2, kernel_size=3, stride=1, padding=1)

				self.moduleThrUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)

				self.moduleTwoNext = torch.nn.Sequential(
					torch.nn.ConvTranspose2d(in_channels=386, out_channels=64, kernel_size=4, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleTwoOut = torch.nn.Conv2d(in_channels=194, out_channels=2, kernel_size=3, stride=1, padding=1)

				self.moduleUpscale = torch.nn.Sequential(
					torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=3, stride=2, padding=1, bias=False),
					torch.nn.ReplicationPad2d(padding=[ 0, 1, 0, 1 ])
				)
			# end

			def forward(self, tensorFirst, tensorSecond, objectInput):
				objectOutput = {}

				tensorInput = objectInput['conv6']
				objectOutput['flow6'] = self.moduleSixOut(tensorInput)
				tensorInput = torch.cat([ objectInput['conv5'], self.moduleFivNext(tensorInput), self.moduleSixUp(objectOutput['flow6']) ], 1)
				objectOutput['flow5'] = self.moduleFivOut(tensorInput)
				tensorInput = torch.cat([ objectInput['conv4'], self.moduleFouNext(tensorInput), self.moduleFivUp(objectOutput['flow5']) ], 1)
				objectOutput['flow4'] = self.moduleFouOut(tensorInput)
				tensorInput = torch.cat([ objectInput['conv3'], self.moduleThrNext(tensorInput), self.moduleFouUp(objectOutput['flow4']) ], 1)
				objectOutput['flow3'] = self.moduleThrOut(tensorInput)
				tensorInput = torch.cat([ objectInput['conv2'], self.moduleTwoNext(tensorInput), self.moduleThrUp(objectOutput['flow3']) ], 1)
				objectOutput['flow2'] = self.moduleTwoOut(tensorInput)

				return self.moduleUpscale(self.moduleUpscale(objectOutput['flow2'])) * 20.0
			# end
		# end

		class Complex(torch.nn.Module):
			def __init__(self):
				super(Complex, self).__init__()

				self.moduleOne = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 2, 4, 2, 4 ]),
					torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleTwo = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 1, 3, 1, 3 ]),
					torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleThr = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 1, 3, 1, 3 ]),
					torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleRedir = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleCorrelation = correlation.ModuleCorrelation()

				self.moduleCombined = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=473, out_channels=256, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFou = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
					torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
				
				self.moduleFiv = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
					torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
				
				self.moduleSix = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
					torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleUpconv = Upconv()
			# end

			def forward(self, tensorFirst, tensorSecond, tensorFlow):
				objectOutput = {}

				assert(tensorFlow is None)

				objectOutput['conv1'] = self.moduleOne(tensorFirst)
				objectOutput['conv2'] = self.moduleTwo(objectOutput['conv1'])
				objectOutput['conv3'] = self.moduleThr(objectOutput['conv2'])

				tensorRedir = self.moduleRedir(objectOutput['conv3'])
				tensorOther = self.moduleThr(self.moduleTwo(self.moduleOne(tensorSecond)))
				tensorCorr = self.moduleCorrelation(objectOutput['conv3'], tensorOther)

				objectOutput['conv3'] = self.moduleCombined(torch.cat([ tensorRedir, tensorCorr ], 1))
				objectOutput['conv4'] = self.moduleFou(objectOutput['conv3'])
				objectOutput['conv5'] = self.moduleFiv(objectOutput['conv4'])
				objectOutput['conv6'] = self.moduleSix(objectOutput['conv5'])

				return self.moduleUpconv(tensorFirst, tensorSecond, objectOutput)
			# end
		# end

		class Simple(torch.nn.Module):
			def __init__(self):
				super(Simple, self).__init__()

				self.moduleOne = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 2, 4, 2, 4 ]),
					torch.nn.Conv2d(in_channels=14, out_channels=64, kernel_size=7, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleTwo = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 1, 3, 1, 3 ]),
					torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleThr = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 1, 3, 1, 3 ]),
					torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFou = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
					torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFiv = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
					torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleSix = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
					torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleUpconv = Upconv()
			# end

			def forward(self, tensorFirst, tensorSecond, tensorFlow):
				objectOutput = {}

				tensorWarp = Backward(tensorInput=tensorSecond, tensorFlow=tensorFlow)

				objectOutput['conv1'] = self.moduleOne(torch.cat([ tensorFirst, tensorSecond, tensorFlow, tensorWarp, (tensorFirst - tensorWarp).abs() ], 1))
				objectOutput['conv2'] = self.moduleTwo(objectOutput['conv1'])
				objectOutput['conv3'] = self.moduleThr(objectOutput['conv2'])
				objectOutput['conv4'] = self.moduleFou(objectOutput['conv3'])
				objectOutput['conv5'] = self.moduleFiv(objectOutput['conv4'])
				objectOutput['conv6'] = self.moduleSix(objectOutput['conv5'])

				return self.moduleUpconv(tensorFirst, tensorSecond, objectOutput)
			# end
		# end

		self.moduleFlownets = torch.nn.ModuleList([
			Complex(),
			Simple(),
			Simple()
		])

		self.load_state_dict(torch.load('./network-' + arguments_strModel + '.pytorch'))
	# end

	def forward(self, tensorFirst, tensorSecond):
		tensorFirst = tensorFirst[:, [ 2, 1, 0 ], :, :]
		tensorSecond = tensorSecond[:, [ 2, 1, 0 ], :, :]

		tensorFirst[:, 0, :, :] = tensorFirst[:, 0, :, :] - (104.920005 / 255.0)
		tensorFirst[:, 1, :, :] = tensorFirst[:, 1, :, :] - (110.175300 / 255.0)
		tensorFirst[:, 2, :, :] = tensorFirst[:, 2, :, :] - (114.785955 / 255.0)

		tensorSecond[:, 0, :, :] = tensorSecond[:, 0, :, :] - (104.920005 / 255.0)
		tensorSecond[:, 1, :, :] = tensorSecond[:, 1, :, :] - (110.175300 / 255.0)
		tensorSecond[:, 2, :, :] = tensorSecond[:, 2, :, :] - (114.785955 / 255.0)

		tensorFlow = None
		#tensorFlowBw = None
		for moduleFlownet in self.moduleFlownets:
			tensorFlow = moduleFlownet(tensorFirst, tensorSecond, tensorFlow)
			#tensorFlowBw = moduleFlownet(tensorSecond, tensorFirst, tensorFlowBw)
		# end

		return tensorFlow
	# end
# end

moduleNetwork = Network().cuda().eval()

##########################################################

def estimate(tensorFirst, tensorSecond):
	assert(tensorFirst.size(1) == tensorSecond.size(1))
	assert(tensorFirst.size(2) == tensorSecond.size(2))

	intWidth = tensorFirst.size(2)
	intHeight = tensorFirst.size(1)

	#assert(intWidth == 1280) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	#assert(intHeight == 384) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	tensorPreprocessedFirst = tensorFirst.cuda().view(1, 3, intHeight, intWidth)
	tensorPreprocessedSecond = tensorSecond.cuda().view(1, 3, intHeight, intWidth)

	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

	tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

	tensorFlow = torch.nn.functional.interpolate(input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	tensorFirstWrapped = Backward(tensorPreprocessedSecond, tensorFlow)

	tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

	return tensorFlow[0, :, :, :].cpu(), tensorFirstWrapped.cpu()
# end

##########################################################

import cv2
import pykitti
import time
from KITTI import KITTI_occ,KITTI_noc

def draw_optical_flow(image,flow,interval):
    for x in range(0,image.shape[0],interval):
        for y in range(0,image.shape[1],interval):
            u,v = flow[:,x,y]
            cv2.arrowedLine(image, (y,x), (int(round(y+u)),int(round(x+v))), (0,0,255), 1)


def image_to_tensor(image):
    return torch.FloatTensor(numpy.array(image.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))


def tensor_to_image(tensor):
	if len(tensor.size())==4:
		tensor = tensor[-1, :, :, :]
	numpy_tensor = tensor.numpy().transpose([1,2,0])
	return numpy.around(numpy_tensor * 255).astype(numpy.uint8)

kitti_dataset = KITTI_occ("/media/liustein/Liustein/Data/data_scene_flow/training")


if __name__ == '__main__':

	# Change this to the directory where you store KITTI data
	basedir = '/media/liustein/Liustein/Data/kitti_odometry'

	# Specify the dataset to load
	sequence = '00'

	# Load the data. Optionally, specify the frame range to load.
	dataset = pykitti.odometry(basedir, sequence)

	index = 0
	image_previous = next(dataset.rgb)[0]
	tensor_previous = image_to_tensor(image_previous[:, :-1, :])
	for image_set in dataset.rgb:
		image = image_set[0][:, :-1, :]

		start = time.time()

		tensor_current = image_to_tensor(image)
		tensor_flow, tensor_first_wrapped = estimate(tensor_previous, tensor_current)
		flow = tensor_flow.data.cpu().numpy()

		end = time.time()
		print((end - start) * 1000)

		first_image_wrapped = tensor_to_image(tensor_first_wrapped)
		ori_wrap = numpy.concatenate([tensor_to_image(tensor_previous), first_image_wrapped, image], axis=0)

		# Obtain the flow magnitude and direction angle
		draw_optical_flow(image, flow, 10)
		cv2.imshow("flow", image)
		cv2.imshow("wrapped", ori_wrap)
		cv2.waitKey(-1)
		tensor_previous = tensor_current

# end