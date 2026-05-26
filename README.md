# Diseño conceptual de un sistema de alerta temprana para la predicción de la pérdida acelerada en el Antonov An-32

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20359823.svg)](https://doi.org/10.5281/zenodo.20359823)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-GUI-41CD52.svg)](https://riverbankcomputing.com/software/pyqt/)

Repositorio oficial del trabajo de grado de Ingeniería Aeronáutica (Centro de Educación Militar - Escuela de Aviación del Ejército, Colombia). Este proyecto implementa un **Banco de Pruebas Unificado** (Testbench) para evaluar un modelo secuencial de aprendizaje automático (`AntonovTransformerModelV3`) diseñado para predecir dinámicas de pérdida acelerada.

## 📌 Descripción General

El sistema actúa como un reproductor de telemetría de alta fidelidad, permitiendo cargar simulaciones previas y evaluar en tiempo real (simulado a 50Hz) el rendimiento del modelo de inferencia. La red neuronal proyecta el estado de la aeronave en un **horizonte futuro de 5 segundos**, calculando el Índice de Riesgo y el *Time-to-Stall* (T2S) con estimación de incertidumbre mediante regresión Tobit.

El entorno y las simulaciones base están configurados estrictamente para las características de rendimiento del Antonov An-32, respetando un Peso Máximo de Despegue (MTOW) operativo de **27,000 kg**.

## 🚀 Características Principales

* **Motor de Inferencia Secuencial (Transformer/LSTM):** Integración de un modelo multicabezal (2 de clasificación, 2 de regresión) que consume series temporales a través de ventanas deslizantes.
* **Alerta Temprana (Horizonte de 5s):** Todas las predicciones, alarmas visuales en el HUD y cálculos de T2S reflejan el estado proyectado de la aeronave a 5 segundos en el futuro, brindando un margen operativo real a la tripulación.
* **Explorador de Datos Crudos (Raw Data):** El sistema se alimenta exclusivamente de las 19 variables físicas en crudo del vector de estado. Se excluyen deliberadamente variables calculadas (como las fuerzas G) para evitar la fuga de datos (*data leakage*) y garantizar la fiabilidad del modelo.
* **Máquina de Estados Físicos (Etiquetador FSM):** El etiquetado temporal sigue una lógica física estricta. La transición a la fase de recuperación (*Estado Cyan*) está matemáticamente bloqueada a menos que la aeronave haya ingresado previamente en una fase de pérdida confirmada (*Estado Red*).
* **Panel de Instrumentos Analógicos (Six-Pack):** Recreación gráfica en PyQt6 de los instrumentos de vuelo tradicionales para un seguimiento visual sincronizado con la telemetría reproducida.

## 📊 Dataset de Vuelo (Zenodo)

Este banco de pruebas consume la base de datos `Base_de_datos_GOLD_Ultimate_PCHIP.h5`, generada a partir de 300 simulaciones de vuelo automatizadas en FlightGear/JSBSim, utilizando Muestreo del Hipercubo Latino (LHS) e interpolación PCHIP para la mitigación del *jitter*.

Para promover la reproducibilidad, el dataset se encuentra en acceso abierto bajo licencia CC BY 4.0:
* **DOI:** [10.5281/zenodo.20359823](https://doi.org/10.5281/zenodo.20359823)
* **Formato:** Archivo jerárquico HDF5, optimizado para lectura iterativa con Pandas.

## ⚙️ Estructura del Repositorio

```text
/
├── 1_Data_Generation/             # (Opcional) Recopilacion de scripts de automatización Nasal, FDM de JSBSim y del pipeline no criticos para el funcionamiento del banco
├── 2_Model_Training/              # Arquitectura y scripts de entrenamiento en PyTorch
└── 3_Testbench/                   # Banco de Pruebas Unificado (Playback)
    ├── core.py                    # Aplicación principal ejecutable (GUI PyQt6)
    ├── modeldriver.py             # Lógica de inferencia y escalado de datos
    ├── panel_sixpack.py           # Agrupación de instrumentos de vuelo
    ├── instrument_widgets.py      # Diales, indicadores y componentes visuales
    ├── check.py                   # Utilidad de validación del dataset HDF5
    ├── data/
    │   └── scalers/               # Objetos de escalado (.pkl) requeridos por el modelo
    └── models/
        └── antonov_v8_produccion.pt  # Pesos finales del modelo (V8)

🎓 Autores
Danna Luna Santa Perilla - Investigadora principal
Santiago Sepulveda Otero - Investigador principal
Director: Daniel Santiago Gutiérrez Pacheco

Centro de Educación Militar - Escuela de Aviación del Ejército (ESAVE). Bogotá D.C., Colombia (2026).
