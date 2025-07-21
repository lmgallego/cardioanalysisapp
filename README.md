# Cycling Performance Analyzer

Una aplicación Streamlit para analizar datos de rendimiento en ciclismo con métricas avanzadas.

## Características

- **Critical Power (CP)**: Cálculo de potencia crítica y capacidad de trabajo anaeróbico (W')
- **rHRI (Relative Heart Rate Increase)**: Análisis del incremento relativo de la frecuencia cardíaca
- **Análisis por Cuartiles de Potencia**: Distribución del tiempo en diferentes zonas de potencia
- **Análisis de Intervalos de 15 Minutos**: Desglose del rendimiento en intervalos de 15 minutos
- **Visualizaciones**: Gráficos detallados para cada tipo de análisis
- **Exportación de Datos**: Descarga de resultados en formato Excel y CSV

## Estructura del Proyecto

```
CardioAnalysis/
├── app.py                  # Aplicación principal de Streamlit
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Documentación del proyecto
└── src/                    # Código fuente modularizado
    ├── data/               # Módulos para carga y procesamiento de datos
    │   ├── __init__.py
    │   └── loader.py       # Funciones para cargar y limpiar datos
    ├── analysis/           # Módulos para análisis de datos
    │   ├── __init__.py
    │   ├── critical_power.py  # Cálculo de potencia crítica
    │   └── rhri.py         # Análisis de rHRI y cuartiles
    ├── visualization/      # Módulos para visualización
    │   ├── __init__.py
    │   └── plots.py        # Funciones para crear gráficos
    └── utils/              # Utilidades generales
        ├── __init__.py
        └── helpers.py      # Funciones auxiliares
```

## Instalación

1. Clona este repositorio:
   ```
   git clone https://github.com/yourusername/CardioAnalysis.git
   cd CardioAnalysis
   ```

2. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

## Uso

1. Ejecuta la aplicación:
   ```
   streamlit run app.py
   ```

2. Sube un archivo CSV con tus datos de ciclismo o utiliza los datos de ejemplo proporcionados.

3. Explora los diferentes análisis disponibles en las pestañas.

## Formato de Datos

El archivo CSV debe contener las siguientes columnas:
- `time`: Tiempo en segundos
- `power` o `watts`: Potencia en vatios
- `heart_rate` o `heartrate`: Frecuencia cardíaca en ppm

## Detalles de los Análisis

### Critical Power (CP)

El modelo de Potencia Crítica identifica:
- **CP**: La potencia que teóricamente puedes mantener indefinidamente
- **W'**: Tu capacidad de trabajo anaeróbico (en julios)
- **R²**: Coeficiente de determinación que indica la calidad del ajuste del modelo

### rHRI (Relative Heart Rate Increase)

El rHRI mide la tasa de incremento de la frecuencia cardíaca en relación a la potencia, proporcionando información sobre:
- Eficiencia cardiovascular
- Fatiga cardíaca durante el ejercicio
- Respuesta del sistema cardiovascular a diferentes intensidades

### Cuartiles de Potencia

Los datos se dividen en cuatro cuartiles basados en el porcentaje de CP:
- **Q1**: < 75% de CP (Recuperación)
- **Q2**: 75-90% de CP (Resistencia)
- **Q3**: 90-105% de CP (Tempo/Umbral)
- **Q4**: > 105% de CP (VO2max/Anaeróbico)

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.
