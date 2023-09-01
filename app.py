import pandas as pd
import streamlit as st
import plotly.express as px
import shap
import pickle
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import API
from zipfile import ZipFile
from API import app


model = pickle.load(open('lgbm_optimized.pkl', 'rb')) 

st.set_page_config(page_title='Analyse du profil client',
                   layout='wide',
                   initial_sidebar_state='expanded')

html_temp = """
<div style="background-color: #74dbce; padding:10px; border-radius:10px">
<h1 style="color: White; text-align:center">Dashboard d'évaluation de solvabilité</h1>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

st.markdown("""
<head>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
</head>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    main = ZipFile("data/main_test.zip")
    df = pd.read_csv(main.open('main_test.csv'))

    pie = ZipFile("data/pie_test.zip")
    df_pie = pd.read_csv(pie.open('pie_test.csv'))
    return df, df_pie

df, df_pie = get_data()

header_left, header_mid = st.columns([1, 3], gap='large')

with st.sidebar:
    st.sidebar.title('Sélectionnez votre identifiant ci-dessous :')

    user_id = st.selectbox(label="Saisissez l'identifiant client :",
                           options=df['SK_ID_CURR'])
    
    st.write('')
            
    check_age = st.checkbox(label='Âge client')
    check_inc = st.checkbox(label='Revenus client')
    check_child = st.checkbox(label="Nombre d'enfants")
    check_income_type = st.checkbox(label="Secteur d'activité")
    check_feat = st.checkbox(label='Impact des paramètres du client')
    
    '''



    '''
        
users_df = df[df['SK_ID_CURR'] == user_id]
users_df = users_df[users_df.columns[1:]]

col1, col2 = st.columns(2)

predict_button = st.sidebar.button('Prédire')

st.sidebar.markdown('''---''')

if predict_button:
    try:
        response = app.post('/predict', json={"user_id": user_id})
    except:
        st.write("Une erreur s'est produite lors de l'appel à l'API.")
        
    if check_age:
        with col1:
            age_bins = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 100]
            age_labels = ['20-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55',
                         '56-60', '61-65', '66-70', '70+']
            
            client_age = np.round(df['DAYS_BIRTH'] / -365)
            client_age_group = pd.cut(client_age, bins=age_bins, labels=age_labels)
            client_age_group = client_age_group.value_counts().reindex(age_labels) 
            
            user_age = np.round(users_df['DAYS_BIRTH']/-365)
            user_age = int(user_age)
            user_age_group = pd.cut([user_age], bins=age_bins, labels=age_labels)[0]
            
            fig = px.histogram(df, x=client_age_group.index, y=client_age_group.values,
                               labels = { 'x' : 'Âge', 'y' : 'Effectifs'}, text_auto=True)
            fig.update_layout(bargap=0.2, title='Répartition des âges de la clientèle :')
            
            user_age_index = age_labels.index(str(user_age_group))
            
            color_sequence = ['indianred' if i != user_age_index else '#fcd444' for i in range(len(age_labels))]
            fig.update_traces(marker=dict(color=color_sequence))

            st.plotly_chart(fig)
            
            st.sidebar.write(f"Âge du client (arrondi à l'entier le plus proche) : {user_age}")

    if check_inc:
        with col2:
            inc_bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650]
            inc_labels = ['0k-50k', '51k-100k', '101k-150k', '151k-200k', '201k-250k', '251k-300k', '301k-350k',
                          '351k-400k', '401k-450k', '451k-500k', '501k-550k', '551k-600k', '601k-650k']
            
            client_inc = df['AMT_INCOME_TOTAL']/1000
            client_inc_group = pd.cut(client_inc, bins=inc_bins, labels=inc_labels)
            client_inc_group = client_inc_group.value_counts().reindex(inc_labels)
            
            user_inc = users_df['AMT_INCOME_TOTAL']/1000
            user_inc = int(user_inc)
            user_inc_group = pd.cut([user_inc], bins=inc_bins, labels=inc_labels)[0]
            
            fig = px.histogram(df, x=client_inc_group.index, y=client_inc_group.values,
                               labels = { 'x' : 'Revenus du client', 'y' : 'Effectifs'}, text_auto=True)
            fig.update_layout(bargap=0.2, title='Répartition des revenus client :')
            
            user_inc_index = inc_labels.index(str(user_inc_group))
            
            color_sequence = ['indianred' if i != user_inc_index else '#fcd444' for i in range(len(inc_labels))]
            fig.update_traces(marker=dict(color=color_sequence))

            st.plotly_chart(fig)
            
            st.sidebar.write(f"Revenus annuels du client : {user_inc}k")

    if check_child:
        with col1:
            fig = px.box(df, x='CNT_CHILDREN', y='AMT_CREDIT',
                         labels = {'CNT_CHILDREN' : "Nombre d'enfants", 'AMT_CREDIT' : 'Montant du crédit accordé'})
            fig.update_layout(title="Montant du crédit accordé en fonction du nombre d'enfants :")
            
            st.plotly_chart(fig)
            
            st.sidebar.write(f"Nombre d'enfants du client : {int(users_df['CNT_CHILDREN'])}")

    if check_income_type:
        with col2:
            user_activity = df_pie[df_pie['SK_ID_CURR'] == user_id]['NAME_INCOME_TYPE'].values[0]

            income_categories = df_pie['NAME_INCOME_TYPE'].unique()

            income_counts = df_pie['NAME_INCOME_TYPE'].value_counts()

            income_effectifs = {category: income_counts.get(category, 0) for category in income_categories}

            fig = px.pie(values=list(income_effectifs.values()), names=income_categories, hole=0.5,
                         color_discrete_sequence=px.colors.qualitative.Dark2)

            fig.update_traces(pull=[0.1 if act == user_activity else 0 for act in income_categories])
            fig.update_layout(title="Secteurs d'activité des clients :")

            st.plotly_chart(fig)

            st.sidebar.write(f"Type de profession du client : {user_activity}")
            
            '''L'activité professionnelle du client correspond au segment disjoint du diagramme.'''
            
    if check_feat:

        fig, ax = plt.subplots(figsize=(10, 10))
        explainer = shap.Explainer(model)

        shap_values = explainer(users_df)

        shap.initjs()
        
        shap.summary_plot(shap_values[:, :, 0], users_df, max_display=20, plot_type="bar", plot_size=(15, 7))

        plt.xlabel('Importances des paramètres lors du calcul du score final', fontsize=8)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=6)

        st.pyplot(fig)
        
    st.sidebar.markdown('''
    ---

    ''')

    
    with st.sidebar:
      predictions = response.json()
      st.write('La probabilité que le client soit solvable est de :', str(("{:.4f}".format(predictions[0])))
      
      if predictions[0] < 0.5:
        st.markdown("""
        <div style="display: flex; align-items: center;">
        <span style="margin-right: 10px;">Éligibilité du client :</span>
        <i class="fas fa-times-circle" style="color:red;"></i>
        </div>""", unsafe_allow_html=True)
      
      else:
        st.markdown("""
        <div style="display: flex; align-items: center;">
        <span style="margin-right: 5px;">Éligibilité du client :</span>
        <i class="fas fa-check-circle" style="color:green;"></i>
        </div>""", unsafe_allow_html=True)
            
      st.write('''
      ---
  
      ''')
            


port = int(os.environ.get("PORT", 8501))
app.run(host='0.0.0.0', port=port)
