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

client = boto3.client('s3')
resource = boto3.resource('s3')
my_bucket = resource.Bucket('capstonedatag58')

def smote_generator(image_comparisons, df, threshold=.25):
    '''
    The SMOTE generator creates synthetic copies of image comparison observations
    in the amount necessary to acheive the desired threshold of dataset balance.

    It outputs the new observations along with the associated image ids and target
    ids for each comparison.
    '''

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
    # print('Beginning create_random_train_images')
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

def comparison_results_to_dataframe(comparisons, images, targets, feature_weights, adjustments=None):

    histogram_comparisons = []
    ssim_comparisons = []
    tweet_match = []
    gradient_similarities = []
    gradient_hash_matches = []

    for comp in comparisons:
        histogram_comparisons.append(comp[0])
        ssim_comparisons.append(comp[1])
        tweet_match.append(comp[2])
        gradient_similarities.append(comp[3])
        gradient_hash_matches.append(4)

    matches = [1 if '_'.join(img.split('_')[0:2]) in tar else 0 for img, tar in zip(images, targets)]

    data = list(zip(images, targets, histogram_comparisons, ssim_comparisons, tweet_match, gradient_similarities, gradient_hash_matches, matches))

    df_comparisons = pd.DataFrame(data=data, columns=['image', 'target', 'histogram', 'ssim', 'tweet_match', 'gradient_similarity', 'gradient_hash', 'matches'])
    df_comparisons['aggregate'] = df_comparisons['histogram'] * feature_weights[0] + df_comparisons['ssim'] * feature_weights[1]

    if adjustments != None:
        df_adjustments = pd.DataFrame(data=adjustments, columns=['image', 'contrast_adjustment'])
        df_comparisons = df_comparisons.merge(df_adjustments, how='left', on='image')

    return df_comparisons

# def cre_train_images(bucket, path, batch_size):
#     s3_image_objects = list(bucket.objects.filter(Prefix=path))[1:]
#     images = []
#     image_ids = []
#
#     file_stream = io.BytesIO()
#
#     for index, obj in enumerate(s3_image_objects):
#         if index % batch_size == 0:
#             yield images, image_ids
#             images = []
#             image_ids = []
#
#         img = read_s3_image_into_opencv(bucket_object, obj.key, file_stream)
#
#         if not cv2.minMaxLoc(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0] != 255: continue
#
#         image_ids.append(obj.key[obj.key.rfind('/')+1:].replace('.jpg', ''))
#         images.append(img)


def load_target_images(bucket_object, path, tweets=True):
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

def load_tp_images(targets, target_ids, number_of_altered_copies, contrast_range=(0.0, 4.0), brightness_range=(0.0, 0.0)):
    tps = targets.copy()
    tp_ids = target_ids.copy()
    adjustments = []

    contrast_params = [i for i in np.linspace(contrast_range[0], contrast_range[1], num=number_of_altered_copies)]
    #brightness_params = [i for i in np.linspace(brightness_range[0], brightness_range[1], num=number_of_altered_copies)]

    for i in range(number_of_altered_copies):
        lab_planes = [cv2.split(cv2.cvtColor(tar, cv2.COLOR_BGR2LAB)) for tar in targets]

        clahe = cv2.createCLAHE(clipLimit=i, tileGridSize=(8,8))

        for lab in lab_planes:
            lab[0] = clahe.apply(lab[0])

        labs = [cv2.merge(lab_plane) for lab_plane in lab_planes]

        transformed_images = [cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) for lab in labs]

        tps.extend(transformed_images)
        tp_ids.extend(list(map(lambda x: x + '_{}'.format(i), target_ids)))
        adjustments.extend(list(zip(list(map(lambda x: x + '_{}'.format(i), target_ids)), [i] * len(target_ids))))

    return tps, tp_ids, adjustments

def show_image_pairs_using_sha1(bucket, path, df, col_1, col_2):
    '''
    View images from a dataframe of results side-by-side by retrieving the
    original images from their S3 buckets.
    '''
    print(path)
    for x, y in zip(df[col_1].values, df[col_2].values):
        print(x)
        print(y)
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
    Retreive an image from S3 and read into opencv.

    Outputs as a numpy array.
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
    Load a specific file from an S3 bucket.
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

def update_urls(data, name, out):
    '''
    Update the appropriate record in the start_urls json file with
    a new start url(s)
    '''

    d = defaultdict(list)
    with open(out, 'r') as f:
        output = json.loads(f.read())

        if list(filter(lambda x: name in x, output)) == []:
            print('Adding new entry into {}'.format(name))
            d[name] = data
            output.append(d)
        else:
            print('Entry already exist, updating entry {}'.format(name))
            for i in output:
                if name in i.keys():
                    i[name] = list(set(i[name] + data))
                    break

    with open(out, 'w') as f:
        json.dump(output, f)


def check_unique_urls(urls):
    '''
    Check a bach of new start urls to ensure they have not been set as a start url
    for a web scrape already.
    '''

    with open('start_urls.json', 'r') as f:
        tracker = json.loads(f.read())

if __name__=='__main__':
    pass
