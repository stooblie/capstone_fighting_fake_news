'''
Modeling.py includes the methods  and workflow necessary for comparing a batch
of images to a set of targets, and formulating a final prediction based on the
comparison results.
'''

#Load utils.py functions and other packages necessary for the analysis.
from utilities import read_s3_image_into_opencv, show_image_pairs_using_sha1, load_images, load_target_images, load_tp_images, comparison_results_to_dataframe, smote_generator
from skimage.measure import compare_ssim as ssim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
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
    The ImageComparison class compares images to a target set and return the comparison
    results.

    Input: Numpy arrays of images.

    Output: A list of tuples including the comparison results for each feature activated
    in the class parameters.
    [(feature1, feature2, etc.), (feature1, feature2, etc.)... ]
    '''

    def __init__(self, histogram=True, size=(96, 96), grayscale=False, equalize=False, ssim=False, tweet_match=False, gradient_similarity=False, gradient_hash=False):
        self.histogram = histogram
        self.grayscale = grayscale
        self.size = size
        self.equalize = equalize
        self.ssim = ssim
        self.tweet_match = tweet_match
        self.gradient_similarity = gradient_similarity
        self.hash = gradient_hash

        self.target_images = None
        self.target_histograms = None

    #Transforms the target images and stores them as an attribute.
    def fit(self, targets):
        self.target_images = self.transform(targets)

        if self.histogram == True:
            self.target_histograms = [cv2.calcHist(images=[tar], channels=[0], mask=None, histSize=[256], ranges=[0,255]) for tar in self.transform(targets)]

    #Transforms the images according to the model parameters.
    def transform(self, images):
        transformed_images = list(map(lambda x: cv2.resize(x, self.size), images))

        if self.grayscale == True:
            transformed_images = list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), transformed_images))

        if self.equalize == True:
            transformed_images = list(map(lambda x: cv2.equalizeHist(x), transformed_images))

        if self.equalize == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

            transformed_images = list(map(lambda x: clahe.apply(x), transformed_images))

        return transformed_images

    #Calculates the hash representation of an image per its gradient direction matrix.
    def calculate_hash(self, image):
        diff = image[:, 1:] > image[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

    #Compares the set of images to the fit targets.
    #Returns a list of tuples.
    def compare(self, images):

        images = self.transform(images)

        if self.hash == True:
            hash_results = []

            for image in images:
                hash_results.extend([hamming(self.calculate_hash(cv2.resize(image, (9,8))), self.calculate_hash(cv2.resize(tar, (9,8)))) for tar in self.target_images])

        else:
            hash_results = [0.] * (len(images) * len(self.target_images))

        if self.histogram == True:
            start = time.clock()
            hist_results = []

            for image in images:
                image_hist = cv2.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0,255])
                hist_results.extend([cv2.compareHist(image_hist, tar, method=0) for tar in self.target_histograms])
            end = time.clock()
        else:
            hist_results = [0.] * (len(images) * len(self.target_images))


        if self.ssim == True:
            start = time.clock()
            ssim_results = []

            for image in images:
                ssim_results.extend([ssim(image, tar) for tar in self.target_images])
            end = time.clock()
        else:
            ssim_results = [0.] * (len(images) * len(self.target_images))

        if self.gradient_similarity == True:
            start = time.clock()
            gradient_results = []

            for image in images:
                flat_image = [item for items in image for item in items]
                gradient_results.extend([1 - cosine(flat_image, [item for items in tar for item in items]) for tar in self.target_images])
            end = time.clock()
        else:
            start = time.clock()
            gradient_results = [0.] * (len(images) * len(self.target_images))
            end = time.clock()


        if self.tweet_match == True:
            start = time.clock()

            tweet_match_results = []

            width = self.size[0] * 0.93

            targets = [cv2.resize(tar, (int(width), int(width * 0.5))) for tar in self.target_images]

            for ind, image in enumerate(images):
                tweet_match_results.extend([cv2.minMaxLoc(cv2.matchTemplate(image, target, cv2.TM_CCOEFF))[1] for target in targets])

            tweet_match_results = np.array(tweet_match_results) / max(tweet_match_results)
            end = time.clock()
        else:
            tweet_match_results = [0.] * (len(images) * len(self.target_images))

        return list(zip(hist_results, ssim_results, tweet_match_results, gradient_results, hash_results))




if __name__=='__main__':
    #Activate S3 connection
    client = boto3.client('s3')
    resource = boto3.resource('s3')
    my_bucket = resource.Bucket('capstonedatag58')

    #Toggle the parameters for usage throughout the comparison and
    #modeling process
    params = {'batch_size': 1000, 'sample_size': 1000, 'model_transform': {'grayscale': True, 'size': (96, 96),
    'equalize': 'clahe'}, 'model_comparison': {'histogram': True, 'ssim': True, 'tweet_match': True, 'gradient_similarity': True, 'gradient_hash': True}, 'tp_adjustments': {'contrast_range': (0.9, 1.1), 'brightness_range': (-10.0, 10.0)},
    'weights': (0.5, 0.5)}

    full_start = time.clock()

    #Load targets from S3 and fit the image class module with them.
    #Elect whther to include tweets images in the target image batch.
    targets, target_ids = load_target_images(my_bucket, 'harvey_scrape/images/target/', tweets=False)
    model = ImageComparison(**params['model_transform'], **params['model_comparison'])
    model.fit(targets)

    #Load the true positive images from S3
    #Elect whether or not to include tweets
    tps, tp_ids = load_target_images(my_bucket, 'harvey_scrape/images/test/', tweets=False)

    #Track the image ids and associated comparisons
    tot_image_ids = []
    comparisons = []
    comp_time = None

    #Run the comparison in batches, loading the images from S3
    #Batching the images uploads with a generator allows the user to manage the
    #memeory usage of the process, should there be constraints
    for images, image_ids in load_images(my_bucket, 'harvey_scrape/images/full/', params['batch_size'], params['sample_size']):
        comp_start = time.clock()
        comp = model.compare(images)
        comp_end = time.clock()
        comp_time = comp_end - comp_start

        tot_image_ids.extend(image_ids)
        comparisons.extend(comp)

    #Create list of image IDs that aligns with the 'comparisons' array
    #There are multiple comparison results per image and the eventual DataFrame \
    #requires a corresponding list of image ids
    tot_image_ids_per_comparison = [tot_image_ids[math.floor(i / len(target_ids))] for i in range(len(comparisons))]

    #Get comparison results for the known true positive images
    #Then, gather lists of true positive and target image ids that correspond
    #with the comparsions (to add into dataframe)
    tp_comparisons = model.compare(tps)

    tp_comparisons_ids = []
    for item in tp_ids:
        tp_comparisons_ids.extend([item] * len(targets))

    tp_comparisons_target_ids = []
    for i in range(int(len(tp_comparisons) / len(target_ids))):
        tp_comparisons_target_ids.extend(target_ids)

    #Filter out the true postive comparison results for the matches between the
    #tp image and target
    #A match will share the first two words in the image name, this will be used
    #as the 'if' condition to determine the match
    tp_matches = [[x, y, z] for x, y, z in zip(tp_comparisons_ids, tp_comparisons_target_ids, tp_comparisons) if '_'.join(x.split('_')[0:2]) == '_'.join(y.split('_')[0:2])]

    #Put the matched images/targets and their comparison results into a dataframe
    tp_df = pd.DataFrame(data=tp_matches, columns=['image', 'target', 'comparison'])

    #Generate synthetic observations from the dataframe of true positive matches
    smote_comparisons, smote_image_ids, smote_target_ids = smote_generator(comparisons, tp_df, threshold=.1)

    #Collect the target image ids asssociated with each true positive comparison
    #This list will be loaded into the final dataframe
    target_ids_full = []
    for i in range(len(tot_image_ids)):
        target_ids_full.extend(target_ids)

    #Construct a dataframe with the image/target and SMOTE comparison results
    df = comparison_results_to_dataframe(comparisons, tot_image_ids_per_comparison, target_ids_full, feature_weights=params['weights'])
    df_smote = comparison_results_to_dataframe(smote_comparisons, smote_image_ids, smote_target_ids, feature_weights=params['weights'])
    df = df.append(df_smote)

    #Create test train splits of the comparisons
    X_train, X_test, y_train, y_test = train_test_split(list(zip(df['histogram'].values, df['ssim'].values, df['tweet_match'].values, df['gradient_similarity'].values, df['gradient_hash'].values)), df['matches'].values)

    #Run the random forest classifier and calculate recall and precision
    rf = RandomForestClassifier(n_estimators=96, max_features=3, max_depth=4)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_recall = recall_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred)
    print('RF Feature Importances: {}'.format(rf.feature_importances_))
    print('RF Model Recall Score: {}'.format(rf_recall))
    print('RF Model Precision Score: {}'.format(rf_precision))

    #Calculate the time taken for the full process
    full_end = time.clock()
    full_time = full_end - full_start
    print('Model time: {}'.format(full_time))

    #Calculate the time taken for the image comparison
    full_time_per_image = full_time / params['sample_size']
    comparison_time_per_image = comp_time / params['sample_size']
    print('Comparison time per image: {}'.format(comparison_time_per_image))
