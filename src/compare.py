'''
Class for comparing scraped images to the targets.
'''

from utilities import read_s3_image_into_opencv, load_images, load_target_images, comparison_results_to_dataframe, smote_generator
from modeling import rf_model
from skimage.measure import compare_ssim as ssim
from scipy.spatial.distance import cosine, hamming
import pandas as pd
import cv2
import pickle
import numpy as np
import boto3
import botocore
import io
import math
import time

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('{}: {} sec'.format(method.__name__, te-ts))
        return result

    return timed

class ImageComparison(object):
    '''
    This class compares images to a target set and returns the comparison results.
    '''

    def __init__(self, size=(96, 96), grayscale=False, equalize=False, histogram=True, ssim=False, tweet_match=False, gradient_similarity=False):
        '''
        Instantiate comparison object.
        :param size: set the pixel dimensions of the images.
        :param grayscale: boolean for converting the images to grayscale.
        :param equalize: boolean for equalizing the images (increase contrast).
        :param histogram: boolean for including a histogram comparison.
        :param ssim: boolean for including ssim in the comparison.
        :param tweet_match: boolean for including the tweet match template in the comparison.
        :param gradient_similarity: boolean for computing the similarity of gradient directions in the comparison.
        '''

        self.histogram = histogram
        self.grayscale = grayscale
        self.size = size
        self.equalize = equalize
        self.ssim = ssim
        self.tweet_match = tweet_match
        self.gradient_similarity = gradient_similarity

        self.target_images = None
        self.target_histograms = None

    def transform(self, images):
        '''
        Transforms the images according to the model parameters.
        '''

        transformed_images = list(map(lambda x: cv2.resize(x, self.size), images))

        if self.grayscale == True:
            transformed_images = list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), transformed_images))

        if self.equalize == True:
            transformed_images = list(map(lambda x: cv2.equalizeHist(x), transformed_images))

        if self.equalize == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

            transformed_images = list(map(lambda x: clahe.apply(x), transformed_images))

        return transformed_images

    def fit(self, targets):
        '''
        Applies transformations to the target images and stores them as an attribute.
        '''
        self.target_images = self.transform(targets)

        if self.histogram == True:
            self.target_histograms = [cv2.calcHist(images=[tar], channels=[0], mask=None, histSize=[256], ranges=[0,255]) for tar in self.transform(targets)]

    def compare(self, images):
        '''
        Compares the set of images to the targets, using the comparison features from the model parameters.

        :param images: The transformed images from the web scrape.
        :return: list, results for comparison feature.
        '''
        images = self.transform(images)

        if self.histogram == True:
            hist_results = []

            for image in images:
                image_hist = cv2.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0,255])
                hist_results.extend([cv2.compareHist(image_hist, tar, method=0) for tar in self.target_histograms])
        else:
            hist_results = [0.] * (len(images) * len(self.target_images))


        if self.ssim == True:
            ssim_results = []

            for image in images:
                ssim_results.extend([ssim(image, tar) for tar in self.target_images])
        else:
            ssim_results = [0.] * (len(images) * len(self.target_images))

        if self.gradient_similarity == True:
            gradient_results = []

            for image in images:
                flat_image = [item for items in image for item in items]
                gradient_results.extend([1 - cosine(flat_image, [item for items in tar for item in items]) for tar in self.target_images])
            gradient_results = gradient_results / max(gradient_results)

        else:
            gradient_results = [0.] * (len(images) * len(self.target_images))


        if self.tweet_match == True:
            tweet_match_results = []

            width = self.size[0] * 0.93

            targets = [cv2.resize(tar, (int(width), int(width * 0.5))) for tar in self.target_images]

            for ind, image in enumerate(images):
                tweet_match_results.extend([cv2.minMaxLoc(cv2.matchTemplate(image, target, cv2.TM_CCOEFF))[1] for target in targets])

            tweet_match_results = np.array(tweet_match_results) / max(tweet_match_results)

        else:
            tweet_match_results = [0.] * (len(images) * len(self.target_images))

        return list(zip(hist_results, ssim_results, tweet_match_results, gradient_results))
