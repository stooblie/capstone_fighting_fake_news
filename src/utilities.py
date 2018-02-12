#The following functions are for the purpose of basic file manipulation.
from scipy.spatial.distance import cosine
from sklearn.neighbors import KNeighborsClassifier
import boto3
import botocore
import json
from collections import defaultdict
import cv2
import numpy as np
import io
import math
import pandas as pd
from time import sleep
import warnings

client = boto3.client('s3')
resource = boto3.resource('s3')
my_bucket = resource.Bucket('capstonedatag58')

def smote_generator(image_comparisons, df, threshold=.25):
    '''
    The SMOTE generator creates synthetic copies of image comparison observations.
    Uses a weighted average of the k-nearest neighbors of the comparison to create the new observation.

    :param image_comparisons: length of the image comparisons list.
    :param df: dataframe of true positive comparison results.
    :param threshold: target proportion of positive targets for rebalancing the classes.
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        tp_comparisons = []
        for comp in df['comparison']:
            tp_comparisons.append([value if value != None else 0. for value in comp])

        d = {str(comp): {'image': img, 'target': tar} for comp, img, tar in zip(tp_comparisons, df['image'], df['target'])}

        kNN = KNeighborsClassifier()
        kNN.fit(tp_comparisons, [1] * len(tp_comparisons))

        synthetic_observations = []
        smote_image_ids = []
        smote_target_ids = []
        while ((len(tp_comparisons) + len(synthetic_observations)) / len(image_comparisons)) < threshold:
            obs = tp_comparisons[np.random.randint(len(tp_comparisons), size=1)]
            obs_id = d[str(obs)]['image'] + '_smote_{}'.format(len(synthetic_observations))
            target_id = d[str(obs)]['target']

            neighbor = tp_comparisons[kNN.kneighbors(obs, 1)[1][0][0]]

            new_obs = []
            for ind, feature in enumerate(obs):
                weight = np.random.random()
                new_feature_value = weight * feature + (1 - weight) * neighbor[ind]
                new_obs.append(new_feature_value)
            synthetic_observations.append(new_obs)
            smote_image_ids.append(obs_id)
            smote_target_ids.append(target_id)

        return synthetic_observations, smote_image_ids, smote_target_ids

def load_images(bucket, path, batch_size, max_quantity):
    '''
    Load the scraped images from their S3 bucket into a list of the array representations of the images.
    Tracks the associated image ids in a separate list.

    :param bucket: S3 bucket object
    :param path: key/file path to the image folder in the bucket.
    :param batch_size: size of the batch of images for the generator to yield.
    :param max_quantity: maximum amount of images to process before stopping.
    :yield: two lists for the images and their associated ids, respectively.
    '''
    sample = np.random.choice(list(bucket.objects.filter(Prefix=path))[1:], size=max_quantity)
    # print('S3 objects loaded in')
    images = []
    image_ids = []

    # print('Establishing file stream')

    file_stream = io.BytesIO()

    # print('File stream set')

    for index, obj in enumerate(sample):
        if (index > 0 and index % batch_size == 0):
            # print('Yielding')
            yield images, image_ids
            images = []
            image_ids = []
            # print('Completed Yield')

        # print('Reading in image')
        img = read_s3_image_into_opencv(bucket, obj.key, file_stream)
        # print('Image read in')
        if not cv2.minMaxLoc(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0] != 255: continue

        # print('Tested blank, appending info to lists')
        image_ids.append(obj.key[obj.key.rfind('/')+1:].replace('.jpg', ''))
        images.append(img)

    yield images, image_ids

def comparison_results_to_dataframe(comparisons, images, targets):
    '''
    Move comparison results into a pandas dataframe.

    :param comparisons: list, the image/target comparison results.
    :param images: list, the images used for the comparison.
    :param targets: list, the targets used for the comparison.
    :return: pandas dataframe with columns for the features, images, targets, and whether it is a match or not.
    '''

    histogram_comparisons = []
    ssim_comparisons = []
    tweet_match = []
    gradient_similarities = []

    for comp in comparisons:
        histogram_comparisons.append(comp[0])
        ssim_comparisons.append(comp[1])
        tweet_match.append(comp[2])
        gradient_similarities.append(comp[3])

    matches = [1 if '_'.join(img.split('_')[0:2]) in tar else 0 for img, tar in zip(images, targets)]

    data = list(zip(images, targets, histogram_comparisons, ssim_comparisons, tweet_match, gradient_similarities, matches))

    df_comparisons = pd.DataFrame(data=data, columns=['image', 'target', 'histogram', 'ssim', 'tweet_match', 'gradient_similarity', 'matches'])

    return df_comparisons


def load_target_images(bucket_object, path, tweets=True):
    '''
    Load target or test images from an S3 bucket.

    :param bucket_object: S3 Object, for the bucket holding the images.
    :param path: str, the key/path for the images in the S3 bucket.
    :param tweets: boolean, whether to include images labeled as tweets in the batch.
    :return: two lists, one for the target images and another with their associated image ids.
    '''
    s3_target_objects = list(bucket_object.objects.filter(Prefix=path))[1:]
    targets = []
    target_ids = []

    for obj in s3_target_objects:
        if (tweets==False and '_tweet_' in obj.key): continue
        file_stream = io.BytesIO()

        target_ids.append(obj.key[obj.key.rfind('/') + 1:].replace('.jpg', ''))

        tar = read_s3_image_into_opencv(bucket_object, obj.key, file_stream)
        targets.append(tar)

    return targets, target_ids

def show_image_pairs_using_sha1(bucket, path, df, col_1, col_2):
    '''
    View two sets of images side by side, grabbing their ids from columns in a pandas dataframe.
    Creates a window that moves to the next pair on any keyboard command then closes.

    :param bucket: S3 Object, bucket where the images are stored.
    :param path: str, key/file path for where the images are stored within the bucket.
    :param df: pandas dataframe including image ids for the images being viewed.
    :param col_1: str, the column of images to view on the lef tof the display.
    :param col_2: str, the column of images to view on the right of the display.
    '''

    for x, y in zip(df[col_1].values, df[col_2].values):
        file_stream = io.BytesIO()
        image_key = path + 'images/full/' + x + '.jpg'
        target_key = path + 'images/target/' + y + '.jpg'

        img_1 = cv2.resize(read_s3_image_into_opencv(bucket, image_key, file_stream), (300, 300))
        img_2 = cv2.resize(read_s3_image_into_opencv(bucket, target_key, file_stream), (300, 300))

        stack = np.hstack((img_1, img_2))
        cv2.imshow('test', stack)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        file_stream.close()

def read_s3_image_into_opencv(bucket, key, io_stream):
    '''
    Retreive an image from S3 and read it into opencv.

    :param : S3 Object, bucket where the images are stored.
    :param key: str, key/file path for the images inside the bucket.
    :param io_stream: io stream object for buffering the images in memory.
    :return: numpy array, representing the image
    '''
    # print('Reading one image into opencv')
    bucket.Object(key).download_fileobj(io_stream)
    # print('Image read into file stream')
    io_stream.seek(0)
    # print('Seek operation completed on image')
    arr = np.asarray(bytearray(io_stream.read()), dtype=np.uint8)
    # print('Image compiled into array, returnning image')
    return cv2.imdecode(arr, 1)

def load_file_from_s3(bucket, path):
    '''
    Retreive a full file from S3.

    :param : S3 Object, bucket where the images are stored.
    :param key: str, key/file path for the file inside the bucket.
    :return: S3 File Object
    '''
    client = boto3.client('s3')
    resource = boto3.resource('s3')
    my_bucket = resource.Bucket(bucket)

    item = json.loads(client.get_object(Bucket=bucket, Key=path)['Body'].read().decode('utf-8'))
    return item

def load_folder_from_s3(bucket, path):
    '''
    Load an entire folder from an S3 bucket.

    Returns an iterable of the S3 objects for the files.
    '''
    client = boto3.client('s3')
    resource = boto3.resource('s3')
    my_bucket = resource.Bucket(bucket)

    return list(my_bucket.objects.filter(Prefix=path))[1:]

def update_urls(urls, name, out):
    '''
    Update the appropriate record in the start_urls json file with
    a new start url(s).

    :param urls: list, the urls to add.
    :param name: str, name of the scrape.
    :param out: str, outfile name.
    '''

    d = defaultdict(list)
    with open(out, 'r') as f:
        output = json.loads(f.read())

        if list(filter(lambda x: name in x, output)) == []:
            print('Adding new entry into {}'.format(name))
            d[name] = urls
            output.append(d)
        else:
            print('Entry already exist, updating entry {}'.format(name))
            for i in output:
                if name in i.keys():
                    i[name] = list(set(i[name] + urls))
                    break

    with open(out, 'w') as f:
        json.dump(output, f)
