import streamlit as st
import plotly.express as px
from sklearn.cluster import DBSCAN
import umap
import numpy as np
import pandas as pd
import ast  # для преобразования строк в списки

# Заголовок приложения
st.title("Пример кластеризации с использованием UMAP и DBSCAN")

# Загружаем данные
uploaded_file = st.file_uploader("Загрузите файл Excel с данными", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Преобразуем строки в массивы, если embeddings хранятся в строковом формате
    df['embeddings'] = df['embeddings'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)

    # Преобразуем столбец 'embeddings' в массив
    embeddings = np.vstack(df['embeddings'].values)

    # Применяем DBSCAN для кластеризации
    dbscan = DBSCAN(eps=0.1, min_samples=5, metric='cosine')
    cluster_labels = dbscan.fit_predict(embeddings)

    # Добавляем метки кластеров в датафрейм
    df['Cluster'] = cluster_labels

    # Уменьшение размерности до 2D для визуализации с использованием UMAP
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    # Добавляем результаты UMAP в датафрейм для визуализации
    df['UMAP1'] = embeddings_2d[:, 0]
    df['UMAP2'] = embeddings_2d[:, 1]

    # Построение интерактивного scatter plot с Plotly
    fig = px.scatter(
        df, x='UMAP1', y='UMAP2',
        color='Cluster',
        hover_data={'Очищенный текст': True, 'UMAP1': False, 'UMAP2': False, 'Cluster': True}
    )

    # Настройка графика
    fig.update_layout(
        xaxis_title="UMAP1",
        yaxis_title="UMAP2"
    )

    # Отображение графика в Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Фильтрация данных и вывод текста по кластерам
    selected_cluster = st.selectbox("Выберите кластер для отображения текста:", sorted(df['Cluster'].unique()))

    # Фильтруем датафрейм по выбранному кластеру и выводим текст
    cluster_texts = df[df['Cluster'] == selected_cluster]['Очищенный текст']
    st.write(f"Тексты для кластера {selected_cluster}:")
    for text in cluster_texts:
        st.write(text)
else:
    st.write("Пожалуйста, загрузите файл с данными для начала.")
