# Detección de fallas en camiones Scania mediante PCA

## Nuestro proyecto

En este proyecto estamos desarrollando una aplicación que nos ayude a identificar si una falla registrada en un camión Scania está relacionada con el sistema de presión de aire, conocido como **APS**.

Queremos trabajar con información de un caso real y entender cómo preparar los datos para que un modelo de aprendizaje automático pueda utilizarlos de manera adecuada.

El resultado se presenta en dos categorías:

- **APS (`pos`)**: la falla registrada está relacionada con el sistema de presión de aire.
- **No APS (`neg`)**: la falla registrada corresponde a otro componente.

La aplicación es una herramienta de análisis académico y su resultado no reemplaza una revisión técnica del vehículo.

## ¿Por qué usamos PCA?

Elegimos el **Análisis de Componentes Principales (PCA)** porque el conjunto de datos tiene muchas variables y queremos estudiar cómo podemos reducirlas conservando parte importante de su información.

Primero organizamos y limpiamos los registros, después aplicamos PCA para representar los datos en **20 componentes principales** y finalmente utilizamos una regresión logística para hacer la clasificación.

Es importante distinguir las dos tareas: **PCA reduce la cantidad de variables y la regresión logística realiza la predicción**.

## Datos con los que trabajamos

Utilizamos el conjunto **APS Failure at Scania Trucks**, publicado en el repositorio UCI Machine Learning Repository.

| Característica | Información |
|---|---|
| Procedencia | Scania, a través de UCI |
| Registros de entrenamiento originales | 60.000 |
| Registros de prueba adicionales | 16.000 |
| Variables de entrada | 170 |
| Tipo de información | Variables numéricas anonimizadas y una categoría de falla |
| Fuente | [Consultar el conjunto de datos en UCI](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks) |

En los archivos originales encontramos valores ausentes identificados como `na`, por eso una parte importante del trabajo consiste en revisar y preparar la información antes del entrenamiento.

Los nombres de las variables están anonimizados, así que no podemos afirmar que cada columna corresponda a un sensor específico.

## ¿Cómo estamos trabajando los datos?

Nuestro proceso está organizado en estas etapas:

1. **Descarga:** obtenemos los archivos originales desde UCI.
2. **Revisión:** comprobamos cuántos registros y variables tenemos, además de los datos faltantes y posibles duplicados.
3. **Separación:** reservamos el 20 % de los datos de entrenamiento para validar los resultados y usamos el 80 % restante para ajustar el modelo.
4. **Limpieza:** identificamos las columnas con demasiados valores ausentes y rellenamos los faltantes de las columnas conservadas con la mediana.
5. **Preparación:** estandarizamos las variables para que sus escalas sean comparables.
6. **PCA:** reducimos las variables a 20 componentes y calculamos cuánta información conservan.
7. **Clasificación:** entrenamos una regresión logística y evaluamos sus resultados.
8. **Aplicación:** utilizamos el modelo guardado para analizar registros desde una interfaz sencilla.

Para revisar la calidad del modelo calculamos métricas como precisión, recall, F1 y accuracy, además de la matriz de confusión.

El conjunto tiene más casos de la categoría `neg` que de `pos`, así que también prestamos atención a las fallas que el sistema podría pasar por alto.

**Los valores de las métricas se obtienen al ejecutar el entrenamiento**, no aparecen como resultados definitivos en este documento.

## ¿Qué hace cada archivo?

Esta es la estructura del repositorio y la función de sus archivos principales:

| Archivo o carpeta | Para qué lo usamos |
|---|---|
| `README.md` | Explica el proyecto, los datos y la forma de ejecutarlo |
| `LICENSE` | Contiene la licencia MIT del repositorio |
| `requirements.txt` | Incluye las bibliotecas de Python necesarias |
| `.gitignore` | Evita subir archivos temporales y datos locales innecesarios |
| `EJECUTAR_PRIMERA_PRUEBA.bat` | Prepara el entorno en Windows, descarga los datos, entrena el modelo y abre la aplicación |
| `INICIAR_EN_WINDOWS.bat` | Abre la aplicación cuando ya ejecutamos la primera prueba |
| `entrenar.py` | Realiza la limpieza dentro del proceso de entrenamiento, aplica PCA, evalúa el modelo y guarda los resultados |
| `scripts/descargar_datos.py` | Descarga los dos conjuntos originales desde UCI |
| `src/preprocessing.py` | Contiene funciones para leer los CSV, revisar valores faltantes y seleccionar columnas |
| `src/__init__.py` | Permite organizar las funciones de apoyo como un módulo de Python |
| `notebooks/01_training.ipynb` | Nos permite explorar los datos y revisar el entrenamiento paso a paso en Jupyter |
| `app/app.py` | Contiene la interfaz de Streamlit para analizar los registros |
| `data/raw/` | Guarda los CSV originales después de descargarlos |
| `data/processed/` | Carpeta reservada para datos procesados |
| `models/` | Guarda el modelo entrenado y el resumen de las métricas después de ejecutarlo |
| `docs/GRUPO_Actividad3_ASESORIA.txt` | Es nuestra plantilla con la información del avance del equipo |

Dentro de `models/` se generan dos archivos importantes al entrenar:

- `modelo_pca_scania.joblib`: guarda el proceso de preparación de datos, PCA y clasificación para volver a utilizarlo.
- `metricas.json`: conserva los resultados obtenidos en la evaluación.

El entrenamiento también genera `docs/INFORME_AVANCE_REAL.txt`, donde podemos consultar los resultados que produjo nuestra propia ejecución.

Las carpetas pueden contener archivos `.gitkeep` para que GitHub las muestre mientras todavía no tienen datos generados.

## ¿Cómo podemos ejecutarlo en Windows?

### Opción sencilla

1. Entramos al [repositorio del proyecto](https://github.com/JuanFer2004/Actividad3).
2. Seleccionamos **Code** y después **Download ZIP**.
3. Descomprimimos el archivo en una carpeta de nuestro computador.
4. Comprobamos que tenemos **Python 3.11** instalado.
5. Abrimos la carpeta y ejecutamos **`EJECUTAR_PRIMERA_PRUEBA.bat`**.
6. Cuando termine el entrenamiento, se abrirá Streamlit en el navegador, normalmente en `http://localhost:8501`.
7. Elegimos uno de los registros disponibles y pulsamos **Analizar registro**.

La primera ejecución necesita conexión a internet para instalar las dependencias y descargar los datos desde UCI.

Si ya ejecutamos el proyecto una vez, podemos abrirlo nuevamente desde **`INICIAR_EN_WINDOWS.bat`**.

### Opción desde la consola

También podemos utilizar la terminal de Windows y ejecutar estos comandos desde la carpeta principal del proyecto:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python scripts/descargar_datos.py
python entrenar.py
python -m streamlit run app/app.py
```

Si la descarga automática falla, podemos obtener los archivos desde la [página oficial de UCI](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks) y ubicarlos en `data/raw/` con los nombres `aps_failure_training_set.csv` y `aps_failure_test_set.csv`.

## ¿En qué etapa vamos?

Ya organizamos el código y la estructura básica del proyecto en GitHub.

Nuestro siguiente paso es ejecutar el entrenamiento en un computador, revisar las métricas que obtengamos y comprobar que la aplicación funciona con el modelo guardado.

Más adelante podremos continuar con el despliegue de la aplicación en la nube.

## Nuestro equipo

Este es un proyecto que desarrollamos en grupo para la asignatura **Inteligencia Artificial I — Actividad 3**.

Cada integrante puede participar en el análisis, revisar el código y registrar sus aportes en el repositorio.

**Integrantes:** Juan Fernando Bueno Torres, Santiago Camacho Parra, Jonathan Pedroza Bernal


## Nuestra aplicación en Streamlit

Desarrollamos una aplicación web que permite analizar registros reales de camiones Scania y clasificarlos según su relación con el sistema de presión de aire (APS).

La aplicación utiliza PCA para reducir las variables y un modelo de regresión logística para realizar las predicciones.

**[Abrir nuestra aplicación en Streamlit](https://actividad3-ngxciwry9uexzaahrfigdl.streamlit.app/)**



## Licencia y fuente

El código del proyecto se comparte bajo la [licencia MIT](LICENSE).

La fuente de los datos es **APS Failure at Scania Trucks**, UCI Machine Learning Repository — DOI: [10.24432/C51S51](https://doi.org/10.24432/C51S51).

Repositorio: https://github.com/JuanFer2004/Actividad3
