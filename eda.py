# eda.py
# Analisis Exploratorio de Datos (EDA) para el dataset de Amazon Reviews.
#
# Genera cuatro visualizaciones que describen la estructura del dataset
# antes de entrenar cualquier modelo:
#   1. Distribucion de ratings (1 a 5 estrellas)
#   2. Distribucion de longitud de resenas (en palabras)
#   3. Scatter de longitud vs. rating
#   4. Boxplot de longitud por cada rating
#
# Uso: python eda.py --data data/amazon_reviews.csv

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Aplica un estilo visual limpio con fondo blanco y paleta de colores suave
sns.set_theme(style="whitegrid", palette="muted")


# -----------------------------------------------------------------------
# CARGA DE DATOS
# -----------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """
    Carga el CSV del dataset y estandariza los nombres de columnas.
    Tambien calcula una columna auxiliar 'word_count' con el numero
    de palabras de cada resena, que se usa en varias graficas.
    """
    df = pd.read_csv(path)

    # Nombres posibles para la columna de texto segun la version del dataset
    text_col   = next(c for c in ["reviewText", "review_text", "Review Text", "text"]
                      if c in df.columns)
    # Nombres posibles para la columna de calificacion objetivo
    target_col = next(c for c in ["overall", "Overall Rating", "rating", "stars"]
                      if c in df.columns)

    # Estandariza los nombres de columna para el resto del script
    df = df.rename(columns={text_col: "review_text", target_col: "overall"})
    df = df.dropna(subset=["review_text", "overall"])
    df["review_text"] = df["review_text"].astype(str)
    df["overall"]     = pd.to_numeric(df["overall"], errors="coerce").clip(1, 5)

    # Cuenta el numero de palabras de cada resena dividiendo por espacios
    df["word_count"]  = df["review_text"].str.split().str.len()

    return df


# -----------------------------------------------------------------------
# GRAFICA 1: DISTRIBUCION DE RATINGS
# -----------------------------------------------------------------------

def plot_rating_distribution(df: pd.DataFrame, out: str):
    """
    Genera un grafico de barras mostrando cuantas resenas hay por cada
    calificacion (1 a 5 estrellas). Permite identificar desbalance de clases,
    que es clave para entender las limitaciones del modelo.
    """
    counts = df["overall"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    # Cada barra tiene un color diferente para resaltar visualmente las categorias
    bars = ax.bar(
        counts.index,
        counts.values,
        color=["#d32f2f", "#f57c00", "#fbc02d", "#388e3c", "#1565c0"]
    )
    ax.set_xlabel("Calificacion (estrellas)", fontsize=12)
    ax.set_ylabel("Numero de resenas",        fontsize=12)
    ax.set_title("Distribucion de Overall Ratings", fontsize=14)
    ax.set_xticks([1, 2, 3, 4, 5])

    # Agrega etiquetas numericas encima de cada barra para lectura directa
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{int(bar.get_height()):,}",
            ha="center", va="bottom", fontsize=10
        )

    fig.tight_layout()
    fig.savefig(os.path.join(out, "rating_distribution.png"), dpi=150)
    plt.close(fig)
    print("  Guardado: rating_distribution.png")


# -----------------------------------------------------------------------
# GRAFICA 2: DISTRIBUCION DE LONGITUD DE RESENAS
# -----------------------------------------------------------------------

def plot_review_length(df: pd.DataFrame, out: str):
    """
    Histograma de la longitud de las resenas en numero de palabras.
    Muestra si la mayoria de resenas son cortas y si hay valores extremos (outliers).
    La linea vertical roja marca la mediana para dar referencia del centro.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(df["word_count"], bins=60, color="steelblue", edgecolor="white", alpha=0.85)

    # Linea vertical en la mediana como referencia del punto central
    mediana = df["word_count"].median()
    ax.axvline(mediana, color="red", linestyle="--", lw=1.5,
               label=f"Mediana: {mediana:.0f} palabras")

    ax.set_xlabel("Palabras por resena", fontsize=12)
    ax.set_ylabel("Frecuencia",          fontsize=12)
    ax.set_title("Distribucion de longitud de resenas", fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "review_length_distribution.png"), dpi=150)
    plt.close(fig)
    print("  Guardado: review_length_distribution.png")


# -----------------------------------------------------------------------
# GRAFICA 3: LONGITUD VS. RATING (SCATTER)
# -----------------------------------------------------------------------

def plot_length_vs_rating(df: pd.DataFrame, out: str):
    """
    Scatter plot que muestra la relacion entre longitud de la resena y su rating.
    Usa una muestra de hasta 2000 puntos para evitar que la grafica quede saturada.
    Cada rating tiene un color diferente para facilitar la lectura.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Toma una muestra aleatoria para que la grafica sea legible
    df_sample = df.sample(min(2000, len(df)), random_state=42)

    for rating, grp in df_sample.groupby("overall"):
        ax.scatter(
            grp["word_count"],
            [rating] * len(grp),   # todos los puntos de este rating en la misma altura
            alpha=0.25,
            s=8,
            label=f"{int(rating)} estrella(s)"
        )

    ax.set_xlabel("Palabras por resena", fontsize=12)
    ax.set_ylabel("Rating",              fontsize=12)
    ax.set_title("Longitud de resena vs. Rating", fontsize=14)
    ax.legend(title="Rating", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "length_vs_rating.png"), dpi=150)
    plt.close(fig)
    print("  Guardado: length_vs_rating.png")


# -----------------------------------------------------------------------
# GRAFICA 4: BOXPLOT DE LONGITUD POR RATING
# -----------------------------------------------------------------------

def plot_boxplot_length_by_rating(df: pd.DataFrame, out: str):
    """
    Boxplot que compara la distribucion de longitudes por cada rating.
    Se excluye el 1% de resenas mas largas (outliers extremos) para que
    las cajas sean visibles y no queden aplastadas por valores atipicos.
    Permite ver si las resenas negativas (1-2 estrellas) tienden a ser
    mas largas o cortas que las positivas (4-5 estrellas).
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Elimina el 1% superior de longitudes para mejorar la visibilidad
    umbral = df["word_count"].quantile(0.99)
    df_plot = df[df["word_count"] < umbral]

    # Agrupa los datos por rating para pasarlos al boxplot
    groups = [df_plot[df_plot["overall"] == r]["word_count"].values
              for r in [1, 2, 3, 4, 5]]

    bp = ax.boxplot(
        groups,
        labels=["1 estrella", "2 estrellas", "3 estrellas", "4 estrellas", "5 estrellas"],
        patch_artist=True,                             # rellena las cajas con color
        medianprops=dict(color="black", lw=2)          # linea de mediana en negro
    )

    # Asigna el mismo color que la grafica de barras para consistencia visual
    colors = ["#d32f2f", "#f57c00", "#fbc02d", "#388e3c", "#1565c0"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Calificacion",        fontsize=12)
    ax.set_ylabel("Palabras por resena", fontsize=12)
    ax.set_title("Longitud de resena por rating (sin el 1% de outliers extremos)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "boxplot_length_by_rating.png"), dpi=150)
    plt.close(fig)
    print("  Guardado: boxplot_length_by_rating.png")


# -----------------------------------------------------------------------
# RESUMEN ESTADISTICO EN CONSOLA
# -----------------------------------------------------------------------

def print_summary(df: pd.DataFrame):
    """
    Imprime en consola un resumen rapido del dataset:
    total de registros, columnas disponibles, distribucion de ratings
    y estadisticas descriptivas de la longitud de resenas.
    """
    print("\n-- Resumen del Dataset ------------------------------------------")
    print(f"  Total de registros : {len(df):,}")
    print(f"  Columnas           : {df.columns.tolist()}")
    print(f"\n  Distribucion de ratings:")
    print(df["overall"].value_counts().sort_index().to_string())
    print(f"\n  Estadisticas de longitud (palabras):")
    print(df["word_count"].describe().round(1).to_string())
    print("-----------------------------------------------------------------\n")


# -----------------------------------------------------------------------
# FUNCION PRINCIPAL
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EDA - Regresor de Estrellas")
    parser.add_argument(
        "--data",
        type=str,
        default="data/amazon_reviews.csv",
        help="Ruta al CSV del dataset de Amazon Reviews"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="images/",
        help="Carpeta donde se guardaran las graficas generadas"
    )
    args = parser.parse_args()

    # Crea la carpeta de imagenes si no existe
    os.makedirs(args.output, exist_ok=True)

    print(f"Cargando datos desde {args.data}...")
    df = load_data(args.data)

    # Muestra un resumen numerico antes de generar las graficas
    print_summary(df)

    # Genera las cuatro visualizaciones del EDA
    print("Generando visualizaciones...")
    plot_rating_distribution(df, args.output)
    plot_review_length(df, args.output)
    plot_length_vs_rating(df, args.output)
    plot_boxplot_length_by_rating(df, args.output)

    print(f"\nEDA completado. Imagenes guardadas en {args.output}/")


if __name__ == "__main__":
    main()
