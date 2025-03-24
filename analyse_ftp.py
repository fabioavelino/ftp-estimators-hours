import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Lecture du fichier CSV
df = pd.read_csv('fabio.csv')

# Préparation des données pour la régression
X = df['hours'].values.reshape(-1, 1)  # Variable indépendante
y = df['ftp'].values  # Variable dépendante

# Création et entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X, y)

# Création de la visualisation
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Données réelles')
plt.plot(X, model.predict(X), color='red', label='Régression linéaire')
plt.xlabel('Heures d\'entraînement')
plt.ylabel('FTP')
plt.title('Relation entre FTP et heures d\'entraînement')
plt.legend()
plt.grid(True)
plt.show()

# Affichage de l'équation de la régression
print(f'Équation de la régression linéaire :')
print(f'FTP = {model.intercept_:.2f} + {model.coef_[0]:.2f} × Heures')