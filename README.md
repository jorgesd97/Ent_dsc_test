# Predicción de Churn — Clientes TELCO
## Challenge Data Scientist — Entel

---

## Descripción del problema

Predecir la fuga (churn) de clientes de una empresa TELCO
a partir de datos de comportamiento de 299.997 clientes
durante 3 meses (Junio, Julio y Agosto 2014).

El objetivo es identificar los 2.000 clientes con mayor
riesgo de churn para una campaña de retención mensual,
maximizando el ROI bajo las siguientes restricciones:

| Parámetro | Valor |
|---|---|
| Contactos por mes | 2.000 |
| Costo por contacto | $1.000 CLP |
| Valor por cliente retenido | $30.000 CLP |
| Tasa de éxito retención | 10% sobre churners reales |
| Umbral de rentabilidad | P@2000 > 33.3% |

---

## Resultado del modelo final

| Métrica | Valor |
|---|---|
| Modelo | Random Forest (Tuned Recall) |
| P@2000 | 37.25% |
| Churners capturados (top 2.000) | 745 / 1.882 |
| Lift vs aleatorio | 19.6x |
| ROI estimado | **+$235.000 CLP** |

---

## Estructura del repositorio
churn-telco/
│
├── data/
│   ├── raw/
│   │   ├── dataset_prueba.csv      ← datos originales
│   │   └── data_dict.csv           ← diccionario de variables
│   └── processed/
│       ├── train_modelo.csv        ← X_train + y_train listos
│       └── test_modelo.csv         ← X_test  + y_test  listos
│
├── models/
│   ├── modelo_rf_tuned_recall_final.pkl  ← modelo ganador
│   ├── modelo_rf_baseline.pkl
│   ├── modelo_xgb_baseline.pkl
│   └── modelo_gbm_baseline.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb                ← análisis exploratorio
│   └── 02_Modelado.ipynb           ← modelamiento y evaluación
│
├── outputs/
│   ├── top2000_campaña.csv         ← clientes a contactar
│   ├── ranking_completo.csv        ← todos los clientes rankeados
│   └── figures/                    ← gráficos generados
│
├── decisiones.md                   ← resumen de decisiones
├── requirements.txt
└── README.md

## Descripción de los notebooks
### 01_EDA.ipynb — Análisis Exploratorio
Contiene el análisis completo de los datos y la preparación
para el modelamiento:

1. Carga y exploración inicial del dataset
2. Análisis de valores nulos y decisión sobre registros NaT
3. Split temporal Train/Test (Jun+Jul → Ago)
4. Segmentación de clientes por ARPU (P25/P75)
5. Análisis univariado y bivariado
6. Clustering RFM con KMeans (4 segmentos)
7. Análisis de churn: patrones y comparativa Churn vs No-Churn
8. Imputación de nulos por semántica de negocio
9. Feature engineering (recency_dias, vbc_efficiency_ratio,
   one-hot encoding de categóricas)
10. Exportación de X_train, y_train, X_test, y_test

**Output:** `train_modelo.csv`, `test_modelo.csv`

---

### 02_Modelado.ipynb — Modelamiento y Evaluación
Contiene el desarrollo del modelo predictivo y el análisis
de negocio:

1. Carga de datos procesados desde EDA
2. Experimento de ratios de desbalance (7 ratios: 1.0 → 0.02)
3. Comparación de 3 modelos: XGBoost, Random Forest, GBM
4. Análisis de 4 escenarios de datos
5. Ensambles (voting simple y ponderado)
6. Feature importance y selección de variables
7. Hyperparameter tuning optimizado por Recall
8. Análisis de negocio: P@2000, ROI, Lift, Gain
9. Priorización de los 2.000 clientes para campaña

**Output:** `modelo_rf_tuned_recall_final.pkl`,
`top2000_campaña.csv`, `ranking_completo.csv`

---

## Cómo ejecutar la solución

### Requisitos
```bash
pip install -r requirements.txt
```

### Orden de ejecución
```bash
# 1. Análisis exploratorio y preparación de datos
jupyter notebook notebooks/01_EDA.ipynb

# 2. Modelamiento y evaluación
jupyter notebook notebooks/02_Modelado.ipynb
```

### En Google Colab
Cada notebook tiene una celda al inicio para montar
Google Drive y cargar los datos desde:
```python
from google.colab import drive
drive.mount('/content/drive')
PATH = '/content/drive/MyDrive/churn-telco/'
```

---

## Dataset

| Archivo | Descripción | Filas | Columnas |
|---|---|---|---|
| dataset_prueba.csv | Datos de clientes | 299.997 | 56 |
| data_dict.csv | Diccionario de variables | 34 | 2 |

**Variables principales:**
- `mobile_number` — Identificador único del cliente
- `last_date_of_month` — Período (Jun/Jul/Ago 2014)
- `arpu` — Ingreso promedio por usuario
- `total_og_mou` — Minutos de llamadas salientes
- `total_ic_mou` — Minutos de llamadas entrantes
- `total_rech_amt` — Monto total de recargas
- `churn` — Variable objetivo (1=fuga, 0=se queda)

---

## Decisiones clave

Ver `decisiones.md` para el resumen completo.

Las más relevantes:
- Split temporal para evitar data leakage
- Formato largo (no pivoteado) para conservar
  el 23.56% de churners de Junio
- Optimización por Recall porque FN es 3x
  más costoso que FP en este negocio
- Métrica de evaluación: P@2000 (no AUC ni F1)

---

## Requisitos
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.8.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0

---

## Autor
Jorge Soto - Challenge Data Scientist Entel