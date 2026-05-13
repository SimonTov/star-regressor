# ⭐ Regresor de Estrellas

Predicción de calificaciones (1–5 estrellas) a partir del texto de reseñas de Amazon usando embeddings de LLM + regresión.

> **Autores:** David Ballesteros · Simón Tovar  
> **Curso:** Machine Learning / AI  
> **Dataset:** [Amazon Review – Kaggle](https://www.kaggle.com/datasets/mehmetisik/amazon-review/data)

---

## Estructura del repositorio

```
star-regressor/
├── data/                        # Dataset (no incluido en el repo)
│   └── amazon_reviews.csv
├── images/                      # Gráficas del EDA
│   ├── rating_distribution.png
│   └── review_length_distribution.png
├── models/                      # Artefactos del modelo entrenado (generados)
│   ├── model.pkl
│   ├── scaler.pkl
│   └── embedder_name.txt
├── results/                     # Gráficas y predicciones de evaluación (generadas)
├── download_data.py             # Descarga el dataset desde Kaggle
├── eda.py                       # Análisis exploratorio de datos
├── train.py                     # Entrenamiento del modelo
├── evaluate.py                  # Evaluación / predicción individual
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
# 1. Clona el repositorio
git clone https://github.com/<tu-usuario>/star-regressor.git
cd star-regressor

# 2. Crea y activa un entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Instala dependencias
pip install -r requirements.txt
```

---

## Obtener el dataset

Necesitas una cuenta en [Kaggle](https://www.kaggle.com) con API key configurada:

1. Ve a **kaggle.com → Settings → API → Create New Token**
2. Guarda el `kaggle.json` descargado en `~/.kaggle/kaggle.json`
3. Ejecuta:

```bash
python download_data.py
```

O descárgalo manualmente desde [este enlace](https://www.kaggle.com/datasets/mehmetisik/amazon-review/data)  
y guárdalo como `data/amazon_reviews.csv`.

---

## Uso

### 1. EDA

```bash
python eda.py --data data/amazon_reviews.csv
```

Genera gráficas en `images/`: distribución de ratings, longitud de reseñas, boxplots.

### 2. Entrenamiento

```bash
# Modelo por defecto: Ridge Regression con embeddings all-MiniLM-L6-v2
python train.py --data data/amazon_reviews.csv

# Alternativas de modelo: ridge | gbr (Gradient Boosting) | rf (Random Forest)
python train.py --data data/amazon_reviews.csv --model gbr

# Usar un embedder diferente (más grande, más lento, posiblemente mejor)
python train.py --data data/amazon_reviews.csv \
    --embedder sentence-transformers/all-mpnet-base-v2
```

El modelo entrenado se guarda en `models/`.

### 3. Evaluación

```bash
# Evaluar sobre el CSV completo
python evaluate.py --data data/amazon_reviews.csv

# Predecir una reseña individual
python evaluate.py --text "This product exceeded all my expectations, absolutely love it!"
```

---

## Enfoque técnico

| Componente | Detalle |
|---|---|
| **Embedder** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| **Regresor** | Ridge Regression (α=1.0) — también disponible GBR y RF |
| **Normalización** | StandardScaler sobre los embeddings |
| **Métricas** | MAE · RMSE |
| **Variable X** | `reviewText` — texto libre de la reseña |
| **Variable y** | `overall` — calificación 1–5 estrellas (continua) |

---

## Resultados esperados

| Métrica | Valor típico (Ridge) |
|---|---|
| MAE  | ~0.65 – 0.80 |
| RMSE | ~1.00 – 1.20 |

> Los valores exactos varían según el modelo seleccionado y la semilla de aleatoriedad.

---

## Referencias

- [LLMs For Regression (arxiv 2024)](https://arxiv.org/abs/2411.14708)
- [Review Rating Regression (Springer 2025)](https://link.springer.com/article/10.1007/s10115-025-02571-7)
- [Rating Prediction using LLM Embeddings (GitHub)](https://github.com/AY-Anish-Yadav/rating_prediction_using_llm_embeddings)
