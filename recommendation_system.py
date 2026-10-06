"""Sistema de recomendación de productos basado en contenido.

Idea: cada producto se describe con características (categoría, marca,
descripción, precio, calificación). Se convierten a vectores numéricos y se
usa un modelo de vecinos más cercanos (similitud coseno) para encontrar los
productos más parecidos a uno dado.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def cargar_datos() -> pd.DataFrame:
    """Crea un conjunto de datos de ejemplo (también podría usarse pd.read_csv)."""
    productos = [
        (1, "Laptop Gamer X1", "Computación", "Asus", "laptop potente para juegos gráficos rgb", 1500, 4.5),
        (2, "Laptop Ultraligera Air", "Computación", "Apple", "laptop ligera para trabajo y estudio batería larga", 1800, 4.7),
        (3, "Laptop Estudiante E5", "Computación", "Lenovo", "laptop económica para estudio y trabajo", 600, 4.0),
        (4, "Mouse Inalámbrico", "Accesorios", "Logitech", "mouse inalámbrico ergonómico para oficina", 25, 4.3),
        (5, "Mouse Gamer RGB", "Accesorios", "Razer", "mouse para juegos con luces rgb y alta precisión", 60, 4.6),
        (6, "Teclado Mecánico", "Accesorios", "Razer", "teclado mecánico para juegos con luces rgb", 110, 4.4),
        (7, "Audífonos Bluetooth", "Audio", "Sony", "audífonos inalámbricos con cancelación de ruido", 200, 4.6),
        (8, "Audífonos Gamer", "Audio", "HyperX", "audífonos con micrófono para juegos y sonido envolvente", 80, 4.2),
        (9, "Parlante Portátil", "Audio", "JBL", "parlante bluetooth portátil resistente al agua", 90, 4.5),
        (10, "Smartphone Pro", "Celulares", "Samsung", "teléfono con cámara avanzada y batería larga", 900, 4.6),
        (11, "Smartphone Básico", "Celulares", "Xiaomi", "teléfono económico con buena batería", 200, 4.1),
        (12, "Tablet Dibujo", "Celulares", "Apple", "tablet para estudio y dibujo con lápiz", 700, 4.7),
    ]
    columnas = ["id", "nombre", "categoria", "marca", "descripcion", "precio", "calificacion"]
    return pd.DataFrame(productos, columns=columnas)


def entrenar_modelo(df: pd.DataFrame):
    """Convierte las características en vectores y entrena el modelo de similitud."""
    preprocesador = ColumnTransformer([
        # Texto -> pesos TF-IDF de las palabras de la descripción
        ("texto", TfidfVectorizer(), "descripcion"),
        # Variables categóricas -> columnas 0/1
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["categoria", "marca"]),
        # Variables numéricas -> escala 0-1 para que ninguna domine
        ("num", MinMaxScaler(), ["precio", "calificacion"]),
    ])
    matriz = preprocesador.fit_transform(df)

    # Vecinos más cercanos usando distancia coseno (menor distancia = más similar)
    modelo = NearestNeighbors(metric="cosine", algorithm="brute")
    modelo.fit(matriz)
    return matriz, modelo


def recomendar(nombre: str, df: pd.DataFrame, matriz, modelo, n: int = 3) -> pd.DataFrame:
    """Devuelve los n productos más similares al producto indicado."""
    coincidencia = df.index[df["nombre"].str.lower() == nombre.lower()]
    if len(coincidencia) == 0:
        raise ValueError(f"Producto no encontrado: {nombre!r}")
    idx = coincidencia[0]

    # Se piden n+1 vecinos porque el primero es el propio producto
    distancias, indices = modelo.kneighbors(matriz[idx], n_neighbors=n + 1)
    resultado = df.iloc[indices[0][1:]][["nombre", "categoria", "marca", "precio"]].copy()
    resultado["similitud"] = (1 - distancias[0][1:]).round(3)
    return resultado.reset_index(drop=True)


if __name__ == "__main__":
    datos = cargar_datos()
    matriz_caracteristicas, modelo_knn = entrenar_modelo(datos)

    for producto in ["Laptop Gamer X1", "Audífonos Bluetooth"]:
        print(f"\nRecomendaciones para '{producto}':")
        print(recomendar(producto, datos, matriz_caracteristicas, modelo_knn))
