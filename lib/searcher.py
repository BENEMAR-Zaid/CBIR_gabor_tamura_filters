import numpy as np
import csv
import cv2

class Search:
	def __init__(self, indexPath):
		self.indexPath = indexPath

	def search(self, queryFeaturesGabor,queryFeaturesTamura):
		results1 = {}
		results2 = {}

		with open(self.indexPath) as f:
			reader = csv.reader(f)

			for row in reader:
				features = [float(x) for x in row[1:]]
				d = self.chi2_distance(features, queryFeaturesGabor)

				results1[row[0]] = d

			f.close()

		#Tamura
		with open("D:\\Work\\Master\\S3\\Analyse, Mining and Indexing\\Flask\\index2.csv") as f:
			reader = csv.reader(f)

			for row in reader:
				features = np.array([np.float32(x) for x in row[1:-1]])
				
				#d2 = np.linalg.norm(features-result_tm)
				d2= self.distance(features,queryFeaturesTamura)
				#d2 = chi2_distance(result_tm,features)
				d2 =  d2 / np.sqrt(3 * 255**2)
				results2[row[0]] = d2

			f.close()

		results1 = sorted([(v, k) for (k, v) in results1.items()])
		results2 = sorted([(v, k) for (k, v) in results2.items()])
		results3 =  results1[0:5] + results2[1:5]

		return results3

	def chi2_distance(self, histA, histB, eps = 1e-10):
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])

		return d

	def distance (self, histA, histB) : 
		distance = cv2.norm(histA, histB, cv2.NORM_L2)
		normalized_distance = distance / np.sqrt(3 * 255**2)
		return normalized_distance