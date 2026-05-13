# train.py
# Entrena el regresor de estrellas usando embeddings de sentence-transformers.
#
# El pipeline es:
#   1. Cargar y limpiar el CSV de Amazon Reviews
#   2. Dividir en conjuntos de entrenamiento y prueba
#   3. Generar embeddings con un modelo de sentence-transformers
#   4. Normalizar los embeddings con StandardScaler
#   5. Entrenar un modelo de regresion (Ridge, GBR o Random Forest)
#   6. Evaluar con MAE y RMSE
#   7. Guardar el modelo, el scaler y las graficas de resultados
#
# Uso basico:
#   python train.py --data data/amazon_reviews.csv
#
# Opciones avanzadas:
#   python train.py --data data/amazon_reviews.csv --model gbr
#   python train.py --data data/amazon_reviews.csv --embedder sentence-transformers/all-mpnet-base-v2

import argparse
import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sentence_transformers import SentenceTransformer


# -----------------------------------------------------------------------
# ARGUMENTOS DE LINEA DE COMANDOS
# -----------------------------------------------------------------------

def parse_args():
    # Define los parametros que el usuario puede pasar al ejecutar el script
    p = argparse.ArgumentParser(description="Regresor de Estrellas - Entrenamiento")

    p.add_argument(
        "--data",
        type=str,
        default="data/amazon_reviews.csv",
        help="Ruta al CSV del dataset de Amazon"
    )
    p.add_argument(
        "--output",
        type=str,
        default="models/",
        help="Carpeta de salida para guardar modelo y scaler"
    )
    p.add_argument(
        "--model",
        type=str,
        default="ridge",
        choices=["ridge", "gbr", "rf"],
        help="Modelo de regresion a usar: ridge | gbr | rf"
    )
    p.add_argument(
        "--embedder",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Nombre del modelo de sentence-transformers para generar embeddings"
    )
    p.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proporcion del dataset que se usara como conjunto de prueba (0.0 a 1.0)"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad de resultados"
    )
    return p.parse_args()


# -----------------------------------------------------------------------
# CARGA Y LIMPIEZA DE DATOS
# -----------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """
    Carga el CSV y estandariza los nombres de columnas.
    El dataset de Kaggle puede tener distintos nombres dependiendo de la version,
    por eso se buscan varios nombres candidatos para las columnas de texto y rating.
    """
    df = pd.read_csv(path)

    # Nombres posibles para la columna de texto de la resena
    text_candidates = ["reviewText", "review_text", "Review Text", "text"]
    # Nombres posibles para la columna de calificacion objetivo
    target_candidates = ["overall", "Overall Rating", "rating", "stars"]

    # Busca cual de los nombres candidatos existe en el dataframe
    text_col = next((c for c in text_candidates if c in df.columns), None)
    target_col = next((c for c in target_candidates if c in df.columns), None)

    # Si no se encuentran las columnas necesarias, lanza un error descriptivo
    if text_col is None or target_col is None:
        raise ValueError(
            f"No se encontraron columnas de texto o target.\n"
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    # Renombra las columnas al nombre estandar que usan el resto de los scripts
    df = df[[text_col, target_col]].rename(
        columns={text_col: "review_text", target_col: "overall"}
    )

    # Elimina filas con valores nulos en las columnas principales
    df = df.dropna(subset=["review_text", "overall"])

    # Limpia espacios en blanco del texto y elimina resenas casi vacias
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"].str.len() > 10]

    # Convierte la columna de rating a numerico y la recorta al rango 1-5
    df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
    df = df.dropna(subset=["overall"])
    df["overall"] = df["overall"].clip(1, 5)

    return df.reset_index(drop=True)


# -----------------------------------------------------------------------
# GENERACION DE EMBEDDINGS
# -----------------------------------------------------------------------

def generate_embeddings(texts: list, model_name: str) -> np.ndarray:
    """
    Carga el modelo de sentence-transformers indicado y genera un embedding
    (vector numerico de representacion semantica) por cada texto de la lista.
    El parametro batch_size controla cuantos textos se procesan a la vez en memoria.
    """
    print(f"  Cargando modelo de embeddings: {model_name}")
    embedder = SentenceTransformer(model_name)

    print(f"  Generando embeddings para {len(texts):,} textos...")
    embeddings = embedder.encode(
        texts,
        batch_size=64,            # procesa 64 textos por iteracion
        show_progress_bar=True,   # muestra barra de progreso en consola
        convert_to_numpy=True,    # devuelve array de numpy, no tensor de PyTorch
    )
    return embeddings


# -----------------------------------------------------------------------
# SELECCION DEL MODELO DE REGRESION
# -----------------------------------------------------------------------

def build_regressor(name: str):
    """
    Devuelve una instancia del modelo de regresion segun el nombre recibido.

    - ridge : Ridge Regression. Rapido y eficiente. Buen punto de partida.
    - gbr   : Gradient Boosting Regressor. Mas lento pero generalmente mas preciso.
    - rf    : Random Forest Regressor. Robusto, paralelizable con n_jobs=-1.
    """
    if name == "ridge":
        # alpha=1.0 es la regularizacion L2 estandar para evitar overfitting
        return Ridge(alpha=1.0)

    if name == "gbr":
        # n_estimators: numero de arboles en el ensemble
        # learning_rate: cuanto contribuye cada arbol nuevo al resultado final
        # max_depth: profundidad maxima de cada arbol de decision
        return GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )

    if name == "rf":
        # n_jobs=-1 usa todos los nucleos disponibles del CPU para paralelizar
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=None,   # sin limite de profundidad, cada arbol crece completamente
            random_state=42,
            n_jobs=-1
        )

    raise ValueError(f"Modelo desconocido: {name}")


# -----------------------------------------------------------------------
# CALCULO DE METRICAS
# -----------------------------------------------------------------------

def evaluate(y_true, y_pred, split="Test"):
    """
    Calcula y muestra MAE y RMSE para un conjunto dado.
    - MAE  (Mean Absolute Error):     error promedio en la misma escala que las estrellas
    - RMSE (Root Mean Squared Error): penaliza mas los errores grandes que el MAE
    Ambas metricas son directamente interpretables en unidades de estrellas (1-5).
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  [{split}] MAE={mae:.4f}  RMSE={rmse:.4f}")
    return {"split": split, "mae": mae, "rmse": rmse}


# -----------------------------------------------------------------------
# GRAFICAS DE RESULTADOS
# -----------------------------------------------------------------------

def plot_results(y_test, y_pred, out_dir):
    """
    Genera y guarda dos graficas en la carpeta de resultados:
    1. Scatter de rating real vs. predicho - muestra que tan cerca esta el modelo
    2. Histograma del error - muestra si el modelo sobreestima o subestima
    """
    os.makedirs(out_dir, exist_ok=True)

    # Grafica 1: Rating real vs. predicho
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.3, s=15, color="steelblue")
    # Linea diagonal roja representa prediccion perfecta (predicho == real)
    ax.plot([1, 5], [1, 5], "r--", lw=1.5, label="Prediccion perfecta")
    ax.set_xlabel("Rating real")
    ax.set_ylabel("Rating predicho")
    ax.set_title("Real vs. Predicho")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "real_vs_predicted.png"), dpi=150)
    plt.close(fig)

    # Grafica 2: Distribucion del error (predicho - real)
    # Un error centrado en 0 indica que el modelo no tiene sesgo sistematico
    errors = y_pred - y_test
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=40, color="coral", edgecolor="white")
    ax.axvline(0, color="black", lw=1.2, linestyle="--")  # linea vertical en cero
    ax.set_xlabel("Error (predicho - real)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribucion del error de prediccion")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_distribution.png"), dpi=150)
    plt.close(fig)

    print(f"  Graficas guardadas en {out_dir}/")


# -----------------------------------------------------------------------
# FUNCION PRINCIPAL
# -----------------------------------------------------------------------

def main():
    args = parse_args()

    # Crea la carpeta de salida si no existe todavia
    os.makedirs(args.output, exist_ok=True)

    # Paso 1: Carga y limpieza de datos
    print("\n[1/5] Cargando datos...")
    df = load_data(args.data)
    print(f"  {len(df):,} registros validos cargados.")
    print(f"  Distribucion de ratings:\n{df['overall'].value_counts().sort_index().to_string()}")

    # Paso 2: Division en conjuntos de entrenamiento y prueba
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["review_text"].tolist(),
        df["overall"].values,
        test_size=args.test_size,
        random_state=args.seed,
    )
    print(f"\n[2/5] Division: {len(X_train_text):,} entrenamiento / {len(X_test_text):,} prueba")

    # Paso 3: Generacion de embeddings para ambos conjuntos
    print("\n[3/5] Generando embeddings...")
    X_train_emb = generate_embeddings(X_train_text, args.embedder)
    X_test_emb  = generate_embeddings(X_test_text,  args.embedder)

    # Normaliza los embeddings para que tengan media=0 y desviacion=1 por dimension.
    # IMPORTANTE: el scaler se ajusta SOLO en train y luego se aplica a ambos conjuntos.
    # Si se ajustara tambien en test se estaria filtrando informacion del futuro.
    scaler = StandardScaler()
    X_train_emb = scaler.fit_transform(X_train_emb)
    X_test_emb  = scaler.transform(X_test_emb)

    # Paso 4: Entrenamiento del modelo de regresion
    print(f"\n[4/5] Entrenando modelo: {args.model}...")
    model = build_regressor(args.model)
    model.fit(X_train_emb, y_train)

    # Paso 5: Evaluacion en ambos conjuntos
    print("\n[5/5] Evaluacion de resultados:")
    train_pred = np.clip(model.predict(X_train_emb), 1, 5)
    test_pred  = np.clip(model.predict(X_test_emb),  1, 5)
    # clip(1,5) garantiza que ninguna prediccion salga del rango valido de estrellas

    train_metrics = evaluate(y_train, train_pred, "Train")
    test_metrics  = evaluate(y_test,  test_pred,  "Test")

    # Guarda las metricas en un archivo JSON para tener registro persistente
    metrics_path = os.path.join(args.output, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump([train_metrics, test_metrics], f, indent=2)

    # Guarda el modelo entrenado con pickle para poder usarlo en evaluate.py
    with open(os.path.join(args.output, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    # Guarda el scaler para que evaluate.py normalice los nuevos textos igual que en train
    with open(os.path.join(args.output, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Guarda el nombre del embedder para que evaluate.py cargue exactamente el mismo
    with open(os.path.join(args.output, "embedder_name.txt"), "w") as f:
        f.write(args.embedder)

    print(f"\n  Modelo guardado en {args.output}/model.pkl")

    # Genera las graficas de evaluacion en la carpeta results/
    plot_results(y_test, test_pred, "results/")

    print("\nEntrenamiento completado.")


if __name__ == "__main__":
    main()
