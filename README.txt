Model files (found on https://github.com/chuanqi305/MobileNet-SSD):
-deploy.prototxt (44 KB)
-mobilenet_iter_73000.caffemodel (23 KB)

Code files:
-final_product.py was run on the Jetson with the Caffe model.
	-Run with command python final_product.py on the Jetson
-car_and_pedestrian.py was run on a Windows machine with CUDA with the Pytorch model.
	-Run with command python car_and_pedestrian.py

Other important files:
-The folder calib_result contains files relevant to stereo vision calibration.
-The video elbanco.mp4 was tested with the Pytorch model and Median Flow tracker.