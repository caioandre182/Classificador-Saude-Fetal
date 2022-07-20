#Manipulação de dados
import pandas as pd
import numpy as np

#Visualização
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.target import FeatureCorrelation

#Machine Learning
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer

#Streamlit 
import streamlit as st 
import time
import pickle

@st.cache(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction']) #two pages

if app_mode == "Home":
    #Título 
    st.markdown("<h1 style='text-align: center; color: grey;'>Classificação de Saúde Fetal</h1>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")
    
    #Taxa de Mortalidade Infantil
    st.write("<h3 style='text-align: left; color: grey;'>Taxa de Mortalidade Infantil</h3>", unsafe_allow_html=True)
    st.write("A taxa de mortalidade infantil é obtida por meio do número de crianças de um determinado local (cidade, região, país, continente) que morrem antes de completar 1 ano, a cada mil nascidas vivas.")
    st.write("Esse dado é um aspecto de fundalmental importância para avaliar a qualidade de vida, por meio dele, é possível obter informações sobre a eficácia dos serviços públicos, tais como: saneamento básico, sistema de saúde, disponibilidade de remédios e vacinas, acompanhamento médico, educação, maternidade, alimentação adequda, entre outros.")
    st.write("Como é de se esperar as regiões mais atingidas pela mortalidade infantil são as com menor poder financeiro. A falta de assistência e de orientação às grávidas e a deficiência na assistência hospital aos recém-nascidos são um dos principais motivos.")

    st.image('img\Infant_Mortality_Rate_World_map.png', caption="Taxa de Mortalidade Infantil Mundial")

    st.markdown("-----------------------------------------------------------------------------")
    #Objetivo
    st.write("<h3 style='text-align: left; color: grey;'>Objetivo</h3>", unsafe_allow_html=True)
    st.write("A ONU espera que, até 2030, os países acabem com as mortes evitáveis de recém-nascidos e crianças menores de 5 anos, reduzindo a mortalidade para pelo menos 25 por 1.000 nascidos vivos.")
    st.write("A mortalidade materna, obviamente, anda junto com a taxa de mortalidade infantil, a OMS calcula que cerca de 830 mulheres morrem todos os dias no mundo, devido a complicações na gravidez. A grande maioria dessas mortes (94%) ocorreu em ambientes de poucos recursos e poderia ter sido evitada.")
    st.write("Diante desse problema, os Cardiotocogramas são uma opção simples e de baixo custo para avaliar a saúde fetal, permitindo que os profissionais de saúde atuem na prevenção da mortalidade infantil e materna. O equipamento funciona da seguinte forma, verifica os batimentos cardíacos e o bem estar do bebê, e é feito com sensores ligados à barriga da gestante que coletam estas informações.")
    st.write("Esse aplicativo tem como objetivo, prever os três possíveis resultados do exame, e assim otimizar e auxiliar os médicos nas tomadas de decisões e conseguindo acompanhar o maior números de pacientes, evitando as mortes que podem e devem ser evitadas.")


if app_mode == "Prediction":
    col1,col2,col3 = st.columns(3)

    with col1:
        st.write('')

    with col2:
        st.image("img\prediction.png")

    with col3:
        st.write('')

    st.write("")

    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Resultado do Exame", type = ['csv','xlsx'])

    if (data_file_1 is not None):
        df = pd.read_csv(data_file_1)

        st.write("<h3 style='text-align: left; color: grey;'>Exame:</h3>", unsafe_allow_html=True)
        st.write("")
        st.write(df)

        st.markdown("-------------------------------------------")

        if st.button("Fazer predição"):
            loaded_model = pickle.load(open('modelo_final.sav', 'rb'))
            prediction = loaded_model.predict(df)
            
            df_resultado = pd.DataFrame(prediction)

            df_resultado = df_resultado.rename(columns={0 : 'Resultado'})

            df_resultado = df_resultado['Resultado'].map({1.0 : 'Normal', 2.0 : 'Suspeito', 3.0 : 'Patológico'})

            with st.spinner('Wait for it...'):
                time.sleep(3)

            st.write(df_resultado)

            st.markdown("-------------------------------------------")
            st.write('Observação:')
            st.write('Quando o modelo foi treinado e testado, teve a acurácia de 98,2%. Apesar da alta precisão, é sempre necessário o auxílio de um médico. ')

            





        

    

    