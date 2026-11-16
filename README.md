# Riesgo académico en Saber Pro: modelo predictivo con CRISP-DM

Proyecto final del curso **Computación 3: Introducción a la Analítica de Datos** · Programa de Ingeniería de Sistemas · Universidad de los Llanos.

**Autores**
- Joshua Esteban Martínez Carvajal (160004319)
- Cristian Camilo Soto Robles (160004838)
- Jesús David Peña Chivatá (160004828)

| Recurso | Enlace |
|---|---|
| Aplicación web (Streamlit) | https://saberpro-riesgo-academico.streamlit.app/ |
| Video de 3 minutos | _ |
| Informe (formato IEEE) | [Consultar PDF](informe/Informe_SaberPro.pdf) |

---

## 1. Descripción del proyecto

Las instituciones de educación superior necesitan identificar con anticipación qué perfiles de estudiantes tienen mayor probabilidad de bajo desempeño en el examen Saber Pro, para focalizar el acompañamiento académico. Este proyecto aplica la metodología **CRISP-DM** (comprensión del negocio, de los datos, preparación, modelado, evaluación y despliegue) sobre los resultados de Saber Pro publicados en Datos Abiertos Colombia.

**Objetivo 1 (predictivo).** Estimar la probabilidad de que un estudiante quede en el **25 % inferior del puntaje compuesto**, usando únicamente variables socioeconómicas, institucionales y del programa. Los puntajes de los módulos no se usan como predictores, para evitar fuga de información (*data leakage*).

**Objetivo 2 (explicativo).** Identificar qué factores pesan más en ese riesgo y verificar si el modelo rinde de forma similar entre subgrupos (género, origen de la institución y estrato).

El modelo se despliega en una aplicación web en la que se ingresa el perfil de un estudiante y se obtiene la probabilidad estimada de riesgo y su nivel (bajo, medio o alto).

## 2. Datos

- **Fuente:** [Resultados únicos Saber Pro](https://www.datos.gov.co/resource/u37r-hjmu), Datos Abiertos Colombia (ID `u37r-hjmu`), con aproximadamente 1,22 millones de registros y 57 columnas.
- **Periodos usados:** 20183, 20195, 20203, 20212, 20222 y 20225. Se excluyeron los periodos con menos de 50.000 registros (convocatorias especiales).
- **Muestra:** 154.984 registros, extraídos por la API de datos.gov.co con un muestreo **estratificado por periodo con asignación proporcional** y bloques aleatorios dentro de cada periodo (semilla 42). La muestra reproduce la población: la diferencia máxima en los porcentajes por categoría de las variables comparadas es de 0,15 puntos porcentuales (`resultados/representatividad_muestra_vs_poblacion.csv`).
- **Puntaje compuesto:** promedio simple de los módulos de razonamiento cuantitativo, lectura crítica, inglés y competencias ciudadanas. Es un proxy propio, no el puntaje global oficial del ICFES.
- **Riesgo:** puntaje compuesto menor o igual al percentil 25 del conjunto de entrenamiento (129,25 puntos).

La muestra no se incluye en este repositorio por su tamaño. El notebook la descarga de la API y la guarda en un archivo local.

## 3. Metodología

| Fase CRISP-DM | Qué se hizo |
|---|---|
| Comprensión del negocio | Definición de objetivos, del riesgo y de los criterios de éxito. |
| Comprensión de los datos | Población por periodo, muestreo, calidad (nulos por variable y periodo) y representatividad. |
| Preparación | Edad derivada de la fecha de nacimiento, puntaje compuesto, partición 80/20 estratificada por periodo, imputación dentro del `Pipeline` ("Sin dato" para categóricas y mediana para la edad) y codificación one-hot. |
| Modelado | Baseline, regresión logística, Random Forest y Gradient Boosting (`HistGradientBoostingClassifier`), con validación cruzada estratificada de 5 particiones. |
| Evaluación | Métricas en el conjunto de prueba, calibración, niveles de riesgo, importancia por permutación y desempeño por subgrupo. |
| Despliegue | Modelo guardado con `joblib` y aplicación en Streamlit. |

## 4. Resultados principales

Muestra de 154.984 registros: 123.987 para entrenamiento y 30.997 para prueba.

**Validación cruzada (entrenamiento)**

| Modelo | ROC-AUC | PR-AUC |
|---|---|---|
| Gradient Boosting (Hist) | 0,780 ± 0,002 | 0,542 ± 0,005 |
| Random Forest | 0,768 ± 0,002 | 0,520 ± 0,006 |
| Regresión logística | 0,766 ± 0,002 | 0,519 ± 0,005 |
| Baseline (clase mayoritaria) | 0,500 ± 0,000 | 0,251 ± 0,000 |

**Conjunto de prueba (Gradient Boosting)**
- ROC-AUC = 0,777 y PR-AUC = 0,528, consistentes con la validación cruzada.
- Con el umbral de clasificación 0,26 (elegido con predicciones fuera de muestra del entrenamiento), la precisión de la clase riesgo es 0,43 y el recall es 0,71. El riesgo base es del 25,1 %.
- **Calibración:** la probabilidad estimada coincide con el riesgo observado en todos los tramos (`resultados/calibracion_por_tramos.csv`, `figs/calibracion.png`). Por ejemplo, en el tramo de probabilidad 0,5–0,6 el riesgo observado fue del 53,7 %.

**Variables más influyentes (importancia por permutación):** departamento de residencia, edad, carácter académico de la institución, núcleo del pregrado y origen de la institución. Las variables del hogar (estrato, educación de los padres) pesan menos una vez se conocen las institucionales. Son asociaciones, no relaciones causales.

**Equidad:** el desempeño es similar entre mujeres y hombres (AUC 0,776 y 0,772). Es algo menor en estratos 1 a 3 y en instituciones privadas (AUC cercano a 0,75) y mayor en estrato 4 y en instituciones oficiales (AUC cercano a 0,80). Los grupos con pocos casos (por ejemplo, "Sin Estrato") deben leerse con cautela.

Los niveles **Bajo / Medio / Alto** de la aplicación conservan los cortes guardados en el artefacto: Bajo si `p < 0.13`, Medio si `0.13 <= p < 0.50` y Alto si `p >= 0.50`. Su verificación en el conjunto de prueba está en `resultados/niveles_de_riesgo_test.csv`.

## 5. Estructura del repositorio

```
├── README.md
├── app.py                     Aplicación Streamlit
├── requirements.txt           Dependencias con la versión con la que se entrenó el modelo
├── modelo_riesgo.joblib       Pipeline entrenado (preprocesamiento + modelo)
├── notebooks/
│   └── Proyecto_SaberPro_CRISPDM.ipynb   Notebook completo, con salidas
├── figs/                      Figuras del informe
├── resultados/                Tablas y métricas en CSV y JSON
└── informe/
    └── Informe_SaberPro.pdf
```

## 6. Cómo reproducirlo

**Notebook (recomendado en Google Colab).** Abre `notebooks/Proyecto_SaberPro_CRISPDM.ipynb` en Colab y ejecuta *Entorno de ejecución → Ejecutar todo*. La descarga de la muestra y la validación cruzada tardan varios minutos. La muestra se guarda en `muestra_saberpro_150k.csv` y se reutiliza si se vuelve a ejecutar.

**Aplicación en local.**

```bash
pip install -r requirements.txt
streamlit run app.py
```

La interfaz está diseñada para escritorio (1366–1920 px). **Evaluar perfil** reúne los 16 predictores en cuatro grupos; las categorías se obtienen del artefacto sin cambiar su codificación. No hay respuestas preseleccionadas y la edad admite enteros entre 16 y 70 años. Al calcular, se conservan las respuestas y se muestra la probabilidad junto con los niveles de riesgo y la referencia histórica de entrenamiento.

Las pestañas **Hallazgos del modelo** y **Acerca del proyecto** presentan las tablas y figuras originales y el contexto académico. Los cambios en un formulario ya calculado requieren pulsar de nuevo **Calcular riesgo** para actualizar la estimación.

La presentación se organiza en `app.py`, `profile_model.py`, `styles.css` y `.streamlit/config.toml`. No se necesitan servicios externos ni dependencias adicionales de interfaz.

**Pruebas de regresión y flujo de Streamlit:**

```bash
python -m unittest discover -s tests -v
```

## 7. Limitaciones y uso responsable

- El puntaje compuesto es un proxy construido con cuatro módulos; no equivale al puntaje global oficial del ICFES.
- "Riesgo" significa quedar en el 25 % inferior de esta muestra, no un nivel absoluto de desempeño.
- Las variables más influyentes son asociaciones. Por ejemplo, la edad probablemente refleja la modalidad de estudio y las condiciones de trabajo, no un efecto de la edad en sí.
- El desempeño del modelo es moderado (ROC-AUC cercano a 0,78) porque solo usa contexto socioeconómico, institucional y del programa.
- Los datos corresponden a seis periodos entre 2018 y 2022; el modelo puede degradarse con periodos posteriores.
- Es una herramienta de apoyo para focalizar acompañamiento. No debe usarse para excluir ni sancionar a estudiantes.

## 8. Tecnologías

Python, pandas, NumPy, scikit-learn, matplotlib, joblib y Streamlit. Entorno de desarrollo: Google Colab.
