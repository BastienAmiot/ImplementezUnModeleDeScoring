# ImplementezUnModeleDeScoring

## Présentation de la problématique

Dans ce projet, "Prêt à dépenser" cherche à mettre en production un modèle capable de déterminer la solvabilité d'un client concernant l'accord d'un crédit à la consommation en fonction de son profil.
Pour se faire, de nombreux paramètres sont mis à disposition.
Les données sont issues de Kaggle (téléchargeables à cette adresse : https://www.kaggle.com/c/home-credit-default-risk/data) et regroupent la totalité des paramètres constituant le profil d'un client en plusieurs datasets.

![image](https://github.com/BastienAmiot/ImplementezUnModeleDeScoring/assets/139744720/2cee76d4-4715-48e4-9427-d8d4f145d8d7)

## Outils et dépendances

Le but du projet est de concevoir un tel modèle (et de l'optimiser), de créer un dashboard résumant le portrait global d'un client pour l'octroi d'un crédit communiquant avec une API qui effectue la prédiction et l'envoi au dashboard, puis de déployer cet ensemble Dashboard + API afin d'y avoir accès depuis le web.

Les principales dépendances nécessaires à la réalisation du projet sont les suivantes :

### Création du modèle :
- Pandas
- Numpy
- Scikit-Learn
- Matplotlib.pyplot
- Lightgbm

### Conception Dashboard - API :
- Streamlit
- Plotly.express
- Shap

## Déploiement :

Deux applications sont créées sur Heroku, une pour l'API et l'autre pour le dashboard. Une fois déployée, il est possible d'utiliser l'interface streamlit du dashboard pour effectuer la prédiction de solvabilité d'un client choisi avec le lien suivant :
https://dashboardscoringcredit-4b3cd19d3108.herokuapp.com/
