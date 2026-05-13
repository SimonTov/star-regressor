# evaluate.py
# Evalua el modelo entrenado sobre un CSV completo, o predice el rating
# de una resena individual que se pasa directamente como texto.
#
# Requiere haber ejecutado train.py antes, ya que carga los archivos
# guardados en la carpeta models/ (model.pkl, scaler.pkl, embedder_name.txt).
#
# Uso:
#   # Evaluar el modelo sobre un CSV completo y ver MAE/RMSE
#   python evaluate.py --data data/amazon_reviews.csv
#
#   # Predecir el rating de una resena individual
#   python evaluate.py --text "This product is amazing, totally worth it!"

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sentence_transformers import SentenceTransformer


# -----------------------------------------------------------------------
# CARGA DE ARTEFACTOS DEL MODELO
# -----------------------------------------------------------------------

def load_artifacts(model_dir: str = "models/"):
    """
    Carga los tres artefactos necesarios para hacer predicciones:
    - model.pkl         : el modelo de regresion entrenado (Ridge, GBR o RF)
    - scaler.pkl        : el StandardScaler ajustado durante el entrenamiento
    - embedder_name.txt : el nombre del modelo de embeddings que se uso en train

    Es fundamental usar exactamente el mismo embedder y el mismo scaler que en
    entrenamiento, de lo contrario las predicciones no tendran sentido.
    """
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    # Lee el nombre del embedder guardado como texto plano
    with open(os.path.join(model_dir, "embedder_name.txt")) as f:
        embedder_name = f.read().strip()

    # Carga el modelo de embeddings en memoria
    embedder = SentenceTransformer(embedder_name)

    return model, scaler, embedder


# -----------------------------------------------------------------------
# PREDICCION DE TEXTOS
# -----------------------------------------------------------------------

def predict_texts(texts: list, model, scaler, embedder) -> np.ndarray:
    """
    Recibe una lista de textos y devuelve un array con la calificacion predicha
    (entre 1.0 y 5.0) para cada uno.

    El proceso es:
      1. Generar embeddings con el mismo modelo que se uso en entrenamiento
      2. Normalizar con el scaler ajustado en entrenamiento
      3. Pasar los embeddings normalizados al modelo de regresion
      4. Recortar las predicciones al rango [1, 5]
    """
    # Genera los embeddings del texto (show_progress_bar solo si hay muchos textos)
    emb = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=len(texts) > 100,
        convert_to_numpy=True
    )

    # Normaliza con el scaler de entrenamiento (solo transform, no fit_transform)
    emb = scaler.transform(emb)

    # Predice y recorta al rango valido de estrellas
    preds = model.predict(emb)
    return np.clip(preds, 1, 5)


# -----------------------------------------------------------------------
# EVALUACION SOBRE UN CSV COMPLETO
# -----------------------------------------------------------------------

def evaluate_csv(path: str, model, scaler, embedder):
    """
    Carga un CSV, genera predicciones para todas las resenas y calcula
    MAE y RMSE comparando con los ratings reales.
    Tambien guarda un CSV con las predicciones y una grafica scatter.
    """
    df = pd.read_csv(path)

    # Detecta los nombres de columna del dataset (puede variar segun version de Kaggle)
    text_col   = next(c for c in ["reviewText", "review_text", "Review Text", "text"]
                      if c in df.columns)
    target_col = next(c for c in ["overall", "Overall Rating", "rating", "stars"]
                      if c in df.columns)

    # Limpieza basica del dataframe
    df = df[[text_col, target_col]].dropna()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 10]
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].clip(1, 5)

    # Genera predicciones para todas las resenas del CSV
    preds  = predict_texts(df[text_col].tolist(), model, scaler, embedder)
    y_true = df[target_col].values

    # Calcula las metricas de evaluacion
    mae  = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    print(f"\nResultados sobre {len(df):,} registros:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")

    # Guarda el CSV con la columna de predicciones agregada
    df["predicted_rating"] = np.round(preds, 2)
    out_path = "results/predictions.csv"
    os.makedirs("results", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Predicciones guardadas en {out_path}")

    # Genera la grafica scatter de rating real vs. predicho
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, preds, alpha=0.3, s=12, color="steelblue")
    # Linea diagonal roja = prediccion perfecta
    ax.plot([1, 5], [1, 5], "r--", lw=1.5, label="Perfecta")
    ax.set_xlabel("Rating real")
    ax.set_ylabel("Rating predicho")
    ax.set_title(f"Evaluacion  |  MAE={mae:.3f}  RMSE={rmse:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/eval_scatter.png", dpi=150)
    plt.close(fig)
    print("  Grafica guardada en results/eval_scatter.png")


# -----------------------------------------------------------------------
# FUNCION PRINCIPAL
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Regresor de Estrellas - Evaluacion")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Ruta al CSV de Amazon Reviews para evaluar el modelo"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Texto de una resena individual para predecir su calificacion"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/",
        help="Carpeta donde estan guardados los artefactos del modelo entrenado"
    )
    args = parser.parse_args()

    # Carga el modelo, scaler y embedder desde disco
    print("Cargando artefactos del modelo...")
    model, scaler, embedder = load_artifacts(args.model_dir)

    if args.text:
        # Modo de prediccion individual: predice el rating de un texto dado
        pred = predict_texts([args.text], model, scaler, embedder)[0]
        # Convierte el numero a una representacion visual de estrellas
        stars = "*" * round(pred)
        print(f"\nResena: \"{args.text}\"")
        print(f"Rating predicho: {pred:.2f}  [{stars}]")

    elif args.data:
        # Modo de evaluacion masiva: evalua sobre todo el CSV
        evaluate_csv(args.data, model, scaler, embedder)

    else:
        # Si no se pasa ninguna opcion, muestra instrucciones de uso
        print("Indica --text <resena> para prediccion individual, o")
        print("       --data <ruta_csv> para evaluar sobre un dataset.")


if __name__ == "__main__":
    main()
