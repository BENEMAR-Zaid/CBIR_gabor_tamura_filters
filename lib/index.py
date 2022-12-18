import os
import cv2
from lib.gabor import GaborDescriptor
from lib.tamura import TamuraDescriptor

def index_all_gabor(params):
	# creating a gaborDescripto instance and its kernels
	gd = GaborDescriptor(params)
	gaborKernels = gd.kernels()
	output_file = 'index.csv'
	c = 1
	all_files = os.listdir('D:\\Work\\Master\\S3\\Analyse, Mining and Indexing\\Flask\\static\\images')  ##path relative to server.py

	#For each image in the database we will extract the Gabor kernels based vector features and saving it in a csv file
	for imagePath in all_files:
		imageId = imagePath[imagePath.rfind("/")+1:]
		image = cv2.imread("D:\\Work\\Master\\S3\\Analyse, Mining and Indexing\\Flask\\static\\images"+imagePath)

		features = gd.gaborHistogram(image,gaborKernels)
		features = [str(f) for f in features]
		# print("c = {}".format(c))
		c += 1
		with open(output_file, 'a', encoding="utf8") as f:
			f.write("%s,%s\n" % ("static/images/"+imageId, ",".join(features)))
			f.close()



def index_one_gabor(imagepath , params):
	# creating a gaborDescripto instance and its kernels
	gd = GaborDescriptor(params)
	gaborKernels = gd.kernels()

	output_file = 'index.csv'
	image = cv2.imread(imagepath)

	# For the uploaded image ,we will extract and return the Gabor kernels based vector features and also saving it in a csv file
	features = gd.gaborHistogram(image,gaborKernels)
	feats = [str(f) for f in features]
	with open(output_file, 'a', encoding="utf8") as f:
		f.write("%s,%s\n" % (imagepath, ",".join(feats)))
		f.close()
	return  features


def index_all_tamura() : 

	tm  = TamuraDescriptor()
	output_file = 'index2.csv'
	all_files = os.listdir('D:\\Work\\Master\\S3\\Analyse, Mining and Indexing\\Flask\\static\\images\\')

	for imagePath in all_files:
		imageId = imagePath[imagePath.rfind("/")+1:]
		image = cv2.imread("D:\\Work\\Master\\S3\\Analyse, Mining and Indexing\\Flask\\static\\images\\"+imagePath)
		features = tm.tamura_filters(image)
		features = [str(f) for f in features]
		with open(output_file, 'a', encoding="utf8") as f:
			f.write("%s,%s,%s\n" % ("static/images/"+imageId, ",".join(features),str(1)))
			f.close()

def index_one_tamura(imagepath) : 

	tm  = TamuraDescriptor()
	output_file = 'index2.csv'
	image = cv2.imread(imagepath)
		
	features = tm.tamura_filters(image)
	feats = [str(f) for f in features]

	with open(output_file, 'a', encoding="utf8") as f:
		f.write("%s,%s,%s\n" % (imagepath, ",".join(feats),str(0)))
		f.close()
	return  features
