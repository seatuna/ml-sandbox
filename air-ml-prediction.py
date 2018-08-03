# import numpy as np
import boto3
import pickle
from io import BytesIO
import requests
import numpy as np
import pandas as pd

'''
JSON Request
{
    "title": "Title of document",
    "type": "Document",
    "likes_cnt": 0,
    "comments_cnt": 1,
    "storyboard_cnt": 1
}
'''

def main(event, context):
    type_document = 0
    type_feedly = 0

    # Set asset type
    if event['type'] == 'Document':
      type_document = 1
    elif event['type'] == 'Feedly':
      type_feedly = 1

    # Create data to be sent to AWS SageMaker
    predict = {
      'likes_cnt': event['likes_cnt'],
      'comments_cnt': event['comments_cnt'],
      'storyboard_cnt': event['storyboard_cnt'],
      'type_document': type_document,
      'type_feedly': type_feedly
    }

    # Load TFIDF pickle and get values for incoming text
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket='air-machine-learning', Key='tfidf.pkl')
    pickled_tfidf = response['Body'].read()
    tfidf_vectorizer = pickle.loads(pickled_tfidf)
    transformed_title = tfidf_vectorizer.transform([event['title']])

    new_df = pd.DataFrame(predict)
    tfidf_df = pd.DataFrame(transformed_title.todense())
    final_df = pd.concat([new_df, tfidf_df], axis=1)

    # send request to Sagemaker
    sagemaker_endpoint = 'https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/kmeans-2018-06-05-14-38-21-961/invocations'
    r = requests.post(sagemaker_endpoint, data={ 'body': final_df.as_matrix().astype(np.float32) })

    print(r)
    return r

    # sagemaker = boto3.client('sagemaker')
    # try:
    #   endpoints = sagemaker.list_endpoints()
    #   print(endpoints)
    # except Exception as e:
    #   print(e)