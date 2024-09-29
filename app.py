import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import boto3
import pandas as pd
import io


from os import listdir
from os.path import isfile, join
from datetime import datetime
import streamlit as st
import pandas as pd
from io import StringIO

import pickle

KAGGLE_DS_MEDICAL_NO_SHOW = 'joniarroba/noshowappointments/data'
KAGGLE_DATASET_NAME = 'gkalpolukcu/knn-algorithm-dataset'
KAGGLE_PATH = 'kaggle-download'
BUCKET_NAME = 'bucket-fiap-tech3-dw'

def GetSessionAWS ():
    session = boto3.Session(aws_access_key_id=var_aws_access_key_id,
        aws_secret_access_key=var_aws_secret_access_key,
        aws_session_token=var_aws_session_token
        )
    s3 = session.resource('s3')
    return s3



def Read_From_AWS (BucketName, BucketPath):    

    s3 = GetSessionAWS()   
    my_bucket = s3.Bucket(BucketName) 

    for my_bucket_object in my_bucket.objects.all():     
        splitedKey = my_bucket_object.key.split("/")

        if splitedKey[0] == BucketPath:
            content_object =  my_bucket_object.get()['Body'].read()           
            #df = pd.read_csv(io.BytesIO(content_object))
            #df.head()
            return content_object


def Read_Bucket_By_Key (BucketName, Key):         
    s3 = GetSessionAWS()   
    my_bucket = s3.Bucket(BucketName)          
    objeto = my_bucket.Object(Key).get()['Body'].read()
    return objeto


st.title('AWS CLI Cloud Access: ')

var_aws_access_key_id = st.text_input("aws_access_key_id", "ASIAYKRRLZFBV3JQ5BXU")
var_aws_secret_access_key = st.text_input("aws_secret_access_key", "9CYFron0dXSocqhSdHvDHKkw770qANWk4dJIa7fj")
var_aws_session_token = st.text_area("aws_session_token", "IQoJb3JpZ2luX2VjEAQaCXVzLXdlc3QtMiJGMEQCICXiLelMY6WBI2sjQuljJtXihH4j+NJDL/+nHDV/cB0qAiAtgL6WaqlxOow7kTROShXVrE+m1XgqoGrlAElBFgZxpCq0AghNEAEaDDU3MjQwNzkyNTA1OSIM3T/0E82f8V2q5ekDKpECqH87QOlaHPlTeVHCsKTDZhvX3d/YfjGT+PxKTZyiFOZDK3X3EyLisp/1RiyJeCv4oaYSuBrfAn+4FLQ6Br2TYwIbZ0lkx+4YeFeIXSOymioYL26UyIdoPo0I0fbNfK4wY77babIqrSkiYXSE1u36xhaAWIeelh8iNcVDbd+XBaZKI6KLWpzAluqW6XPoQ4FL6EinKp49L5NLRsq2mNHlvzs/I5m4WZoHcHuFLhxkbz+Ciehg/Km0qRy+J00ZSoDky31JwYu21b74mZPzXT9L8K3MRCKUpbyyO6ogKK+tJT2deoOyGeLINtMkDdjBWskIiSP7cg4QdZIcARtvbth3waz5ApdcRlz0UcMjNytW5QA3MKy84bcGOp4B2nvSflkxB3/5Ufz+sJwq5VDpbPeMujGFag4ngUNFY4kJ6JgtGC93ajrbdX0mrRZcn6HEFn8hvzXmJc31a1If7G0Qh/iY1kA8Zax96stLgSEwDJB1OOXOzQH2g/PCQnXlN1L6xWMR/Oip94vG8MXzVfW9cOvgIBr+h32a40Sv/SJRt2ROd9bA+m7S9usqUvoL5oG22z/+IRQn/0Qtm2o=")

var_Bucket_Name = st.text_input("Bucket_Name", "bucket-fiap-tech3-dw")
var_Bucket_Path = st.text_input("Bucket_Path", "ML-MEDICAL-NOSHOW")
var_pkl_File = st.text_input("PKL File", "pipeline.pkl ")


st.title('Carregue seu arquivo aqui')

uploaded_file = st.file_uploader("Choose a file")


if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    #st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    #st.write(dataframe)

    st.dataframe(dataframe, use_container_width=True)

aws_path = 'ML-MEDICAL-NOSHOW'
kaggle_ds = 'joniarroba/noshowappointments'

if st.button("Predição", type="primary", use_container_width=False):
    st.write("Baixando Peakle da AWS para inicio da Predição")

    object = Read_From_AWS(BUCKET_NAME, var_Bucket_Path)
    df = pd.read_csv(io.BytesIO(object))
    df.head()
      
    st.write("Dataframe Shape :" +str(df.shape))
    st.dataframe(df, hide_index=False)
