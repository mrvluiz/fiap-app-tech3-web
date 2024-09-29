import pandas as pd
import boto3
import pandas as pd
import io
import streamlit as st

import streamlit as st

from io import StringIO
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


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
    return io.BytesIO(objeto)



st.title('AWS CLI Cloud Access: ')

var_aws_access_key_id = st.text_input("aws_access_key_id", "ASIAYKRRLZFBYYRZJUPI")
var_aws_secret_access_key = st.text_input("aws_secret_access_key", "bhGH5xd34o3ivF2oVGBf9L6AmzhHW7V0oXe1hJYk")
var_aws_session_token = st.text_area("aws_session_token", "IQoJb3JpZ2luX2VjEA0aCXVzLXdlc3QtMiJHMEUCIQD13mKXzF+nOTQAvFKIyT6ibaiTS6Vo/zAFaIdfpO2mLQIgOWx3kwvTSZsbYPkyjIhjvtdbB5T5BNP4NTiBag3mG+IqtAIIVhABGgw1NzI0MDc5MjUwNTkiDBFHVX+4zdThkzFHNiqRAtRmfSuyuHy3lpQlGHoE6EWDpbm3LfINTkV6Tgn3enlkECLS2Yk+9cZlgBCfAI9+/z2LXI9TZM7d73dJFaRzy6QMcOYpdDF8/ITV5u9ucU7FJGmNYtJcUMzt+4fSM8b8lXJrrtmiwQtVQmcxqmcYKMmiBMxDIGOj9LIwYGXLZzlB+1O+CkYx6h8UyCytZaz+kB9jccRZyb0j9WeGSg/wJAlmtUIgToLeN0V/q7lMd21KhgHYZZuhCKA9LvH8gHzsuTSTtVV+NIVBRcbXvCNo/Kk2jHuZ9beSDuiplFmA0svWvLtacqVHmhP1Q4omfSQUxk2p9U1hDN6iwEjXQRs8FcTvnlXuwKyLKWsXMeN9gb7YvjCqveO3BjqdAaHpY++YwU7HDFjKt2KFcN0LU8m2nMAy0RWwyg49a/8dxpngOkeNfqO0DFGTnawC5nrulZ3V62AinRXu+oP9TkohksSk0SA1eWoup8rAzpHsHihTVag5MwfQPeeLPk6I6XTrFb938u6Js4cvdrFpo9QsM1NdBfmqJZMZQs1m3AebZd8dpur8RthF5uofkZA0ASO4hw0T9i+Cp31KZ7A=")

var_Bucket_Name = st.text_input("Bucket_Name", "bucket-fiap-tech3-dw")
var_Bucket_Path = st.text_input("Bucket_Path", "ML-MEDICAL-NOSHOW")
var_pkl_File = st.text_input("PKL File", "pipeline.pkl ")


st.title('Carregue seu arquivo aqui')

uploaded_file = st.file_uploader("Choose a file")

dataframe = None

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

    if dataframe is None: 
        st.write("Selecione o arquivo para predição !! ")   
    else:        
        st.write("Baixando Peakle da AWS para inicio da Predição")
        s3Pickle = Read_Bucket_By_Key(var_Bucket_Name, var_pkl_File )
        pipeline = pickle.load(s3Pickle)      
        st.write("Exibindo informações do Peakle da AWS")
        cnt = 0
        for item in pipeline:
            print('The data ', cnt, ' is : ', item)
            cnt += 1

        st.write("tratando dados do CSV")

        dataframe.insert(loc=3, column='HorasAteDataDaConsulta', value = None)
        dataframe = dataframe.drop('PatientId', axis=1)
        dataframe = dataframe.drop('AppointmentID', axis=1)
        dataframe['HorasAteDataDaConsulta'] = (pd.to_datetime(dataframe['ScheduledDay']) - pd.to_datetime(dataframe['AppointmentDay'])).dt.total_seconds() / 3600
        dataframe['ScheduledDay'] = pd.to_numeric(pd.to_datetime(dataframe['ScheduledDay']))
        dataframe['AppointmentDay'] = pd.to_numeric(pd.to_datetime(dataframe['AppointmentDay']))
        
        st.write("Dataframe Shape :" +str(dataframe.shape))
        st.dataframe(dataframe, hide_index=False)


        # Selecione as features relevantes para a predição
        X_validacao = dataframe[['ScheduledDay', 'AppointmentDay', 'HorasAteDataDaConsulta',
                                'Age', 'Scholarship', 'Hipertension', 'Diabetes',
                                'Alcoholism', 'Handcap', 'SMS_received', 'Gender', 'Neighbourhood']]

        # Faça a predição usando o pipeline treinado
        y_pred = pipeline.predict(X_validacao)

        # Adicione as predições ao DataFrame de validação
        dataframe['Predição'] = y_pred
        dataframe.head()

        st.write("Execução Concluída :" +str(dataframe.shape))
        st.dataframe(dataframe, hide_index=False)

        chart_data = dataframe[[ 'Predição']]

        st.bar_chart(chart_data)


