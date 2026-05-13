# download_data.py
# Descarga automaticamente el dataset de Amazon Reviews desde Kaggle.
#
# Kaggle acepta credenciales en dos formatos:
#
#   FORMATO NUEVO (access_token) - el que probablemente tienes:
#     Archivo: C:\Users\TuNombre\.kaggle\access_token
#     Contenido: KGAT_xxxxxxxxxxxxxxxx  (solo el token, sin comillas)
#
#   FORMATO ANTIGUO (kaggle.json):
#     Archivo: C:\Users\TuNombre\.kaggle\kaggle.json
#     Contenido: {"username": "tu_usuario", "key": "tu_clave"}
#
# Si no tienes ninguno, ve a kaggle.com -> Settings -> API -> Create New Token
#
# Uso: python download_data.py

import os
import sys

# -----------------------------------------------------------------------
# VERIFICACION Y CONFIGURACION DE CREDENCIALES
# -----------------------------------------------------------------------

kaggle_dir         = os.path.expanduser("~/.kaggle")
access_token_path  = os.path.join(kaggle_dir, "access_token")
kaggle_json_path   = os.path.join(kaggle_dir, "kaggle.json")

# Si existe el archivo access_token (formato nuevo), lo convierte al formato
# kaggle.json que la libreria de Python espera encontrar.
# Esto es necesario porque la libreria kaggle==1.6.17 aun no lee access_token.
if os.path.exists(access_token_path) and not os.path.exists(kaggle_json_path):
    print("Detectado archivo access_token. Configurando credenciales...")

    with open(access_token_path, "r") as f:
        token = f.read().strip()

    # El token de Kaggle tiene el formato KGAT_xxxxx
    # La API lo acepta con username="__token__" y key=<el token>
    import json
    kaggle_json = {"username": "__token__", "key": token}

    with open(kaggle_json_path, "w") as f:
        json.dump(kaggle_json, f)

    # En Mac/Linux el archivo debe tener permisos restringidos (600)
    # En Windows este paso no es necesario pero no causa error
    try:
        os.chmod(kaggle_json_path, 0o600)
    except Exception:
        pass

    print("Credenciales configuradas correctamente.")

# Verifica que exista alguna forma de autenticacion antes de continuar
tiene_json    = os.path.exists(kaggle_json_path)
tiene_env     = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
tiene_token   = bool(os.getenv("KAGGLE_API_TOKEN"))

if not (tiene_json or tiene_env or tiene_token):
    print("No se encontraron credenciales de Kaggle.")
    print("Crea el archivo de token en:")
    print(f"  {access_token_path}")
    print("O el archivo JSON en:")
    print(f"  {kaggle_json_path}")
    sys.exit(1)

# -----------------------------------------------------------------------
# DESCARGA DEL DATASET
# -----------------------------------------------------------------------

from kaggle.api.kaggle_api_extended import KaggleApi

os.makedirs("data", exist_ok=True)

api = KaggleApi()
api.authenticate()

print("Descargando dataset 'mehmetisik/amazon-review' desde Kaggle...")
print("Esto puede tardar unos segundos dependiendo de la conexion.")

api.dataset_download_files(
    "mehmetisik/amazon-review",
    path="data/",
    unzip=True,
    quiet=False
)

# -----------------------------------------------------------------------
# RENOMBRADO AL NOMBRE ESTANDAR
# -----------------------------------------------------------------------

# Renombra el CSV descargado al nombre que usan los otros scripts
for fname in os.listdir("data/"):
    if fname.endswith(".csv") and "amazon" in fname.lower():
        src = os.path.join("data", fname)
        dst = os.path.join("data", "amazon_reviews.csv")
        if src != dst:
            os.rename(src, dst)
            print(f"Archivo renombrado: {fname} -> amazon_reviews.csv")
        break

print("Dataset listo en data/amazon_reviews.csv")
print("Ahora puedes ejecutar: python eda.py")
