import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
import umap
import pandas as pd

# Подготовьте данные для визуализации
# Предполагаем, что df - это датафрейм с колонками 'Очищенный текст', 'Cluster', и 2D-координатами UMAP1 и UMAP2

# Пример данных для теста, если требуется
# df = pd.DataFrame({"Очищенный текст": ["Текст1", "Текст2"], "Cluster": [0, 1], "UMAP1": [1, 2], "UMAP2": [3, 4]})

# Заголовок приложения

st.title("Интерактивная визуализация кластеров текстов")

df = pd.read_excel('data.xlsx')

# Построение интерактивного графика с Plotly
fig = px.scatter(
    df, x='UMAP1', y='UMAP2',
    color='Cluster',
    hover_data={'Очищенный текст': True, 'UMAP1': False, 'UMAP2': False, 'Cluster': True}
)

# Отображение графика в Streamlit
st.plotly_chart(fig, use_container_width=True)
