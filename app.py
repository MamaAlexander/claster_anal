import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
import umap
import pandas as pd

# Заголовок приложения
st.title("Интерактивная визуализация кластеров текстов")

# Загружаем данные
df = pd.read_excel('data.xlsx')

# Убедитесь, что у вас есть векторные представления текста
# Пример использования колонки 'embeddings' для снижения размерности
if 'embeddings' in df.columns:
    embeddings = pd.DataFrame(df['embeddings'].tolist())  # Преобразуем колонку списков в DataFrame

    # Применяем UMAP для снижения размерности до 2D
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)

    # Добавляем результаты снижения размерности в датафрейм
    df['UMAP1'] = umap_embeddings[:, 0]
    df['UMAP2'] = umap_embeddings[:, 1]

    # Проверяем наличие колонки с кластерами
    if 'Cluster' not in df.columns:
        # Используем DBSCAN для кластеризации, если кластеры не определены
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
        df['Cluster'] = dbscan.fit_predict(embeddings)
else:
    st.error("В датафрейме отсутствуют эмбеддинги для снижения размерности и визуализации.")
    st.stop()

# Построение интерактивного графика с Plotly
fig = px.scatter(
    df, x='UMAP1', y='UMAP2',
    color='Cluster',
    hover_data={'Очищенный текст': True, 'UMAP1': False, 'UMAP2': False, 'Cluster': True}
)

# Отображение графика в Streamlit
st.plotly_chart(fig, use_container_width=True)
