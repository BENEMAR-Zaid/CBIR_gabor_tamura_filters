import numpy as np
import cv2

class TamuraDescriptor : 

    def __init__(self):
        return
    
    def tamura_filters(self,image):
        
        # Convert the images to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Compute the Tamura texture features
        eigenValsandVecs = cv2.cornerEigenValsAndVecs(gray, blockSize=3, ksize=3)
        eigenvals = eigenValsandVecs[:, :, :2]
        eigenvecs = eigenValsandVecs[:, :, 2:]
        # Replace zero or negative values in the eigenvalue arrays with a small positive value
        eigenvals[eigenvals <= 0] = 1e-6
        eigenvecs[eigenvecs <= 0] = 1e-6
        # Compute the coarseness of the images
        coarseness = (eigenvals[:,0] / eigenvals[:,1])
        # Compute the contrast of the images
        contrast = (eigenvals[:,0] / eigenvals[:,2])
        # Compute the directionality of the images
        directionality = (eigenvecs[:,0] / eigenvecs[:,1])
        # Create feature vectors for the images
        features = [np.mean(coarseness), np.mean(contrast), np.mean(directionality)]
        
        feature_vector = np.array(features)
        

        return feature_vector

