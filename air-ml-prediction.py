# import numpy as np
import boto3
import pickle
from io import BytesIO

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

    # Create matrix to be sent to AWS SageMaker
    matrix = [[
      event['likes_cnt'],
      event['comments_cnt'],
      event['storyboard_cnt'],
      type_document,
      type_feedly
    ]]

    # Load TFIDF pickle and get values for incoming text
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket='air-machine-learning', Key='tfidf.pkl')
    pickled_tfidf = response['Body'].read()
    tfidf_vectorizer = pickle.loads(pickled_tfidf)
    transformed_title = tfidf_vectorizer.transform(event['title'])

    print('matrix: \n', matrix)
    print('tfidf: \n', tfidf_vectorizer)
    print('transformed title: \n', transformed_title)

    # sagemaker = boto3.client('sagemaker')
    # try:
    #   endpoints = sagemaker.list_endpoints()
    #   print(endpoints)
    # except Exception as e:
    #   print(e)