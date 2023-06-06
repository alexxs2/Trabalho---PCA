# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 22:18:18 2023

@author: cesar 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize_data(data):
    min_value = np.min(data)
    max_value = np.max(data)
    normalized_data = (data - min_value) / (max_value - min_value)
    return normalized_data

df = pd.read_csv('C:/Users/YungBeta/Desktop/codigo.csv')
selected_columns = df[['Overall rank', 'Year', 'Score', 'GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']]
X = selected_columns.to_numpy()

normalized_data = normalize_data(X)

cov_matrix = np.cov(normalized_data, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
largest_eigenvectors = eigenvectors[:, sorted_indices[:2]]

projected_data = normalized_data.dot(largest_eigenvectors)

plt.scatter(projected_data[:, 0], projected_data[:, 1], c=df['Score'], cmap='coolwarm', alpha=0.5)
plt.colorbar(label='Score')
plt.xlabel('Primeira Componente Principal')
plt.ylabel('Segunda Componente Principal')
plt.title('PCA com 2 Componentes Principais')

important_columns = selected_columns.columns[sorted_indices[:2]]
plt.annotate(important_columns[0], (0.1, 0.9), xycoords='axes fraction')
plt.annotate(important_columns[1], (0.1, 0.85), xycoords='axes fraction')

plt.tight_layout()
plt.savefig('pca_scatterplot.jpg', dpi=300)
plt.show()

plt.subplot(2, 1, 1)
plt.hist(projected_data[:, 0], bins=20, alpha=0.5, color='red')
plt.xlabel('Primeira Componente Principal')
plt.ylabel('Frequência')
plt.title('Histograma da Primeira Componente Principal')

plt.subplot(2, 1, 2)
plt.hist(projected_data[:, 1], bins=20, alpha=0.5, color='blue')
plt.xlabel('Segunda Componente Principal')
plt.ylabel('Frequência')
plt.title('Histograma da Segunda Componente Principal')

plt.tight_layout()
plt.savefig('pca_histogram.jpg', dpi=300)
plt.show()
