import os
import boto3
from updateIndexs import VECTOR_LOCAL
from tools import *

def main():
    s3 = boto3.client('s3')
    upload_folder(s3, VECTOR_LOCAL)
    
if __name__ == '__main__':
    main()