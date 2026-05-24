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

REQUISITOS PREVIOS
----------------------------------------------------------------
- Python 3.10 o superior instalado
- Cuenta en Kaggle con API token configurado
- Conexion a internet (para descargar el dataset y los embeddings)
- La carpeta del proyecto debe ser:
    C:\Users\SIMON\Downloads\star-regressor\star-regressor



PASO 0 — ABRIR LA TERMINAL EN LA CARPETA CORRECTA
=

1. Abrir PowerShell o CMD
2. Navegar a la carpeta del proyecto:

   cd C:\Users\SIMON\Downloads\star-regressor\star-regressor

3. Verificar que estan los archivos correctos:

   dir

   Debes ver: train.py, evaluate.py, eda.py, download_data.py,
              requirements.txt, README.md


PASO 1 — INSTALAR LAS DEPENDENCIAS
=

Ejecutar una sola vez. Si ya lo hiciste antes, puedes saltarlo.

   pip install -r requirements.txt

Esto instala: sentence-transformers, scikit-learn, pandas,
numpy, matplotlib, seaborn, kaggle y sus dependencias.
Puede tardar 2-5 minutos dependiendo de la conexion.

Si aparece un aviso de "new release of pip available", ignorarlo.
No afecta el funcionamiento.


PASO 2 — CONFIGURAR EL TOKEN DE KAGGLE
=

Solo necesitas hacerlo una vez. Si ya lo hiciste, saltalo.

2a. Ve a kaggle.com -> tu foto -> Settings -> API
    -> Create New Token  (o usa el que ya creaste)

2b. Se descarga un archivo. El token tiene el formato:
    KGAT_xxxxxxxxxxxxxxxxxxxxxx

2c. Abrir PowerShell y ejecutar estos dos comandos:

    New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle"

    "TU_TOKEN_AQUI" | Out-File -FilePath "$env:USERPROFILE\.kaggle\access_token" -Encoding ascii

    (Reemplaza TU_TOKEN_AQUI con tu token real, incluyendo las comillas)

2d. Verificar que se creo el archivo:

    dir "$env:USERPROFILE\.kaggle"

    Debes ver el archivo access_token listado ahi.



PASO 3 — DESCARGAR EL DATASET
=

   python download_data.py

Que hace:
- Detecta el token de Kaggle automaticamente
- Descarga el dataset Amazon Review desde Kaggle
- Lo guarda como data/amazon_reviews.csv

Tiempo estimado: 1-3 minutos segun la conexion.

Si aparece el mensaje "Detectado archivo access_token.
Configurando credenciales..." es normal, significa que
convirtio el token al formato que necesita la libreria.

Al terminar debe aparecer:
"Dataset listo en data/amazon_reviews.csv"


PASO 4 — CORRER EL ANALISIS EXPLORATORIO (EDA)
=

   python eda.py

Que hace:
- Lee el CSV descargado
- Imprime un resumen del dataset en consola
- Genera 4 graficas y las guarda en la carpeta images/

Graficas generadas:
- images/rating_distribution.png      (distribucion de estrellas)
- images/review_length_distribution.png  (longitud de resenas)
- images/length_vs_rating.png         (scatter longitud vs rating)
- images/boxplot_length_by_rating.png (boxplot por rating)

Tiempo estimado: 1-2 minutos.


PASO 5 — ENTRENAR EL MODELO
=

Opcion A — Modelo basico (recomendado para empezar):

   python train.py

Opcion B — Especificar modelo diferente:

   python train.py --model gbr
   python train.py --model rf

Opcion C — Usar un embedder mas grande (mas lento, posiblemente
           mejor):

   python train.py --embedder sentence-transformers/all-mpnet-base-v2

Que hace:
- Carga y limpia el CSV
- Divide en 80% entrenamiento / 20% prueba
- Descarga el modelo de embeddings (solo la primera vez,
  luego queda en cache)
- Genera los embeddings para todos los textos
- Entrena el regresor Ridge (u otro si se especifica)
- Imprime MAE y RMSE en consola
- Guarda el modelo en models/model.pkl
- Guarda las graficas en results/

ADVERTENCIA: El paso de generar embeddings es el mas lento.
Con 1.4 millones de resenas puede tardar entre 10 y 40 minutos
dependiendo del hardware. Es normal que la barra de progreso
tarde en avanzar al principio.

Al terminar debe aparecer:
"[Test] MAE=0.xxxx  RMSE=1.xxxx"
"Entrenamiento completado."


PASO 6 — EVALUAR EL MODELO
=

Opcion A — Predecir una resena individual:

   python evaluate.py --text "This product is amazing, works perfectly!"

   Imprime el rating predicho con su representacion en asteriscos.
   Ejemplo de salida: "Rating predicho: 4.73  [*****]"

Opcion B — Evaluar sobre todo el CSV:

   python evaluate.py --data data/amazon_reviews.csv

   Calcula MAE y RMSE sobre el dataset completo y guarda las
   predicciones en results/predictions.csv


RESUMEN DE COMANDOS EN ORDEN
=

cd C:\Users\SIMON\Downloads\star-regressor\star-regressor
pip install -r requirements.txt
python download_data.py
python eda.py
python train.py
python evaluate.py --text "Great product, works perfectly!"


ESTRUCTURA DE CARPETAS DESPUES DE CORRER TODO
=

star-regressor/
├── data/
│   └── amazon_reviews.csv        <- dataset descargado
├── images/
│   ├── rating_distribution.png   <- graficas del EDA
│   ├── review_length_distribution.png
│   ├── length_vs_rating.png
│   └── boxplot_length_by_rating.png
├── models/
│   ├── model.pkl                 <- modelo entrenado
│   ├── scaler.pkl                <- normalizador
│   ├── embedder_name.txt         <- nombre del embedder usado
│   └── metrics.json              <- MAE y RMSE guardados
├── results/
│   ├── real_vs_predicted.png     <- graficas de evaluacion
│   ├── error_distribution.png
│   └── predictions.csv           <- predicciones (si se corre opcion B)
├── download_data.py
├── eda.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md



ERRORES COMUNES Y SOLUCION
=

ERROR: No such file or directory: requirements.txt
  -> Estas en la carpeta equivocada. Ejecuta:
     cd C:\Users\SIMON\Downloads\star-regressor\star-regressor

ERROR: No se encontraron credenciales de Kaggle
  -> El token no esta configurado. Repite el Paso 2.

ERROR: 403 Forbidden al descargar el dataset
  -> El token es invalido o expiro. Ve a Kaggle y genera uno nuevo.
     Repite el Paso 2c con el token nuevo.

ERROR: ModuleNotFoundError: No module named 'sentence_transformers'
  -> Las dependencias no estan instaladas. Ejecuta el Paso 1.

ERROR: FileNotFoundError: models/model.pkl
  -> Todavia no has entrenado el modelo. Ejecuta primero el Paso 5.

La barra de progreso de embeddings se quedo quieta
  -> Es normal. Esta procesando en batches. Espera, no interrumpas.

================================================================

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
