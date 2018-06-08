import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import boto3

'''
JSON Request
{
  title: 'Title of document',
  type: 'Document',
  'likes_cnt': 0,
  'comments_cnt': 1,
  'storyboard_cnt': 1
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

  # TO DO: Load TFIDF pickle and get values for incoming text
  
  # sagemaker = boto3.client('sagemaker')
  # try:
  #   endpoints = sagemaker.list_endpoints()
  #   print(endpoints)
  # except Exception as e:
  #   print(e)