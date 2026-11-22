"""Interfaz local para clasificar registros de camiones Scania usando PCA."""
from pathlib import Path
import json
import sys
import joblib
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.preprocessing import leer_scania

st.set_page_config(page_title='PCA | Fallas de camiones Scania', page_icon='🚚', layout='wide')
st.title('Identificación de fallas APS en camiones Scania')
st.caption('Prototipo académico| PCA + clasificador logístico')
st.info('Las variables son indicadores operativos anonimizados por Scania. El resultado es una clasificación estadística, no un diagnóstico mecánico.')

MODELO = RAIZ / 'models' / 'modelo_pca_scania.joblib'
META = RAIZ / 'models' / 'metricas.json'
TEST = RAIZ / 'data' / 'raw' / 'aps_failure_test_set.csv'
if not MODELO.exists() or not META.exists():
    st.error('Todavía no hay modelo entrenado. Ejecuta: python scripts/descargar_datos.py y luego python entrenar.py')
    st.stop()
if not TEST.exists():
    st.error('Faltan los datos reales de prueba. Ejecuta python scripts/descargar_datos.py')
    st.stop()


@st.cache_resource
def cargar_modelo():
    return joblib.load(MODELO)


@st.cache_data
def cargar_test():
    return leer_scania(TEST)


modelo = cargar_modelo()
metadata = json.loads(META.read_text(encoding='utf-8'))
datos = cargar_test()

c1, c2, c3, c4 = st.columns(4)
c1.metric('Casos reales de prueba', f'{len(datos):,}')
c2.metric('Variables originales', metadata['variables_antes'])
c3.metric('Componentes PCA', metadata['componentes_pca'])
c4.metric('Varianza conservada', f"{metadata['varianza_explica_pca']:.1%}")

st.subheader('1. Selecciona un registro REAL del conjunto de prueba de UCI')
positivos = datos.index[datos['class'] == 'pos'].tolist()
negativos = datos.index[datos['class'] == 'neg'].tolist()
opciones = {
    'Ejemplo A — fallo no relacionado con APS': negativos[0],
    'Ejemplo B — fallo relacionado con APS': positivos[0],
}
tipo = st.selectbox('Registro que deseas comprobar', list(opciones))
indice = opciones[tipo]
fila = datos.loc[[indice]].drop(columns=['class']).copy()

st.subheader('2. Modifica tres indicadores numéricos (opcional)')
st.caption('Valores iniciales tomados del registro real. Cambiarlos crea un escenario de prueba, NO una observación original del dataset.')
columnas = [c for c in metadata['columnas_utilizadas'] if pd.notna(fila.iloc[0][c])][:3]
if len(columnas) < 3:
    st.warning('Este registro tiene menos de tres indicadores numéricos disponibles.')
else:
    cols = st.columns(3)
    for cont, nombre in zip(cols, columnas):
        with cont:
            valor_original = float(fila.iloc[0][nombre])
            nuevo_valor = st.number_input(
                f'Indicador {nombre}', value=valor_original, format='%.4f', key=f'{indice}_{nombre}'
            )
            fila[nombre] = nuevo_valor

if st.button('🔎 Analizar registro', type='primary', use_container_width=True):
    prediccion = int(modelo.predict(fila)[0])
    confianza = float(modelo.predict_proba(fila)[0, 1])
    real = datos.loc[indice, 'class']
    if prediccion == 1:
        st.error('Resultado del modelo: falla relacionada con APS (clasificación positiva)')
    else:
        st.success('Resultado del modelo: falla no relacionada con APS (clasificación negativa)')
    st.metric('Puntuación estimada de clase APS', f'{confianza:.1%}')
    st.write(f'**Etiqueta conocida del registro de prueba original:** {"APS" if real == "pos" else "No APS"}')
    st.caption('Si modificaste indicadores, la etiqueta real corresponde al registro original, no a los valores modificados.')
    st.caption('La puntuación es una salida del modelo; no se ha calibrado como probabilidad de falla real.')

with st.expander('¿Qué limpieza de datos se hizo?'):
    r = metadata['dataset_original']
    st.write(f"Se detectaron **{r['nulos_total']:,}** celdas sin dato y **{r['duplicados']:,}** filas duplicadas completas.")
    st.write('Se descartaron las columnas con ≥50 % de ausencias, usando solo entrenamiento. Luego se imputaron medianas, se estandarizó y se aplicó PCA.')
    st.dataframe(
        pd.DataFrame(list(r['nulos_por_variable'].items()), columns=['Variable', 'Nulos']),
        hide_index=True
    )

with st.expander('Resultados de la validación del modelo'):
    met = metadata['metricas_validacion']
    st.write({k: round(v, 4) if isinstance(v, float) else v for k, v in met.items()})
    st.caption('El accuracy aislado puede ser engañoso por el desbalance de clases. También se informa recall, precisión y costo de errores.')

st.divider()
st.caption('Fuente: UCI Machine Learning Repository, APS Failure at Scania Trucks (dataset 421).')
