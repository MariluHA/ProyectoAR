# Clasificador de Productividad Rural con Machine Learning

## Descripción del Proyecto

Aplicación web desarrollada en Python con Flask que utiliza algoritmos de Machine Learning para clasificar organizaciones rurales según su nivel de productividad. El sistema permite seleccionar entre tres modelos diferentes (Random Forest, SVM y XGBoost) para realizar predicciones basadas en variables internas y externas de la organización.

La aplicación utiliza un dataset estático embebido en el código, sin necesidad de base de datos externa. Los datos de productividad se normalizan y categorizan mediante percentiles para crear las etiquetas de clasificación: Alta, Media y Baja productividad.

## Estructura del Proyecto

```
rural-productivity-classifier/
├── app.py                 # Backend con Flask y lógica de ML
├── requirements.txt       # Dependencias del proyecto
├── Procfile               # Configuración para Render
├── runtime.txt            # Versión de Python (opcional)
└── templates/
    └── index.html         # Interfaz web con formulario
```

## Requisitos del Sistema

- Python 3.9 o superior
- Pip para gestión de paquetes
- Navegador web moderno con soporte para JavaScript

## Instalación Local

Para ejecutar la aplicación en tu máquina local, sigue estos pasos:

1. Clona o descarga el repositorio en tu computadora.
2. Crea un entorno virtual de Python para aislar las dependencias del proyecto.
3. Activa el entorno virtual y navega a la carpeta del proyecto.
4. Instala las dependencias ejecutando el comando `pip install -r requirements.txt`.
5. Inicia la aplicación con el comando `python app.py`.
6. Abre tu navegador web y visita la dirección `http://localhost:5000` para acceder a la aplicación.

## Uso de la Aplicación

La interfaz de usuario presenta un formulario organizado en tres secciones principales. En la sección de configuración, puedes seleccionar el modelo de Machine Learning que deseas utilizar para la predicción. Las opciones disponibles son Random Forest Classifier, Support Vector Machine y XGBoost Classifier.

La sección de variables internas contiene los indicadores propios de la organización, como el índice de desarrollo organizacional, el nivel educativo promedio, la composición por género y el tipo de producto principal que cultiva o produce la organización.

La sección de variables externas abarca los factores del entorno que pueden influir en la productividad, incluyendo el tiempo de ejecución del plan, la brecha territorial (que indica el nivel de acceso a infraestructura y servicios), y las condiciones climáticas como precipitación, temperatura y nivel de sequía.

Una vez completados todos los campos, pulsa el botón «Predecir Productividad» para obtener el resultado de la clasificación. El sistema mostrará el nivel de productividad predicho junto con un resumen de los datos ingresados y el modelo utilizado.

## Despliegue en Render

Render es una plataforma de alojamiento en la nube que permite desplegar aplicaciones web de forma gratuita y sencilla. A continuación, se detallan los pasos para desplegar la aplicación en Render.

### Preparación del Repositorio

1. Crea una cuenta en Render si aún no tienes una.
2. Sube los archivos del proyecto a un repositorio en GitHub o GitLab.
3. Asegúrate de que la estructura de archivos sea la siguiente: el archivo `app.py` debe estar en la raíz del repositorio, junto con `requirements.txt` y `Procfile`. La carpeta `templates` debe contener el archivo `index.html`.

### Creación del Servicio Web en Render

1. Inicia sesión en tu cuenta de Render y accede al dashboard.
2. Haz clic en el botón «New» y selecciona «Web Service» en el menú desplegable.
3. Conecta tu cuenta de GitHub o GitLab y selecciona el repositorio que contiene el proyecto.
4. En la configuración del servicio, especifica los siguientes valores:
   - **Name:** `rural-productivity-classifier` (o el nombre que prefieras)
   - **Branch:** `main` o `master` (dependiendo de tu rama principal)
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Haz clic en «Create Web Service» para iniciar el despliegue.

### Verificación del Despliegue

Una vez completado el proceso de construcción, Render mostrará el estado del servicio. Si todo ha funcionado correctamente, verás un enlace URL en la sección «Your web service is live». Haz clic en ese enlace para acceder a la aplicación desplegada.

Si el despliegue falla, revisa los logs de construcción en Render para identificar el error. Los problemas más comunes suelen estar relacionados con versiones incompatibles de paquetes o errores de sintaxis en el código.

## Modelos de Machine Learning

### Random Forest Classifier

El modelo Random Forest es un algoritmo de conjunto que construye múltiples árboles de decisión durante el entrenamiento. Para esta aplicación, se configura con 100 estimadores y una profundidad máxima de 5 niveles. Este modelo es robusto ante el sobreajuste y maneja bien las relaciones no lineales entre variables.

### Support Vector Machine (SVM)

El clasificador SVM busca encontrar el hiperplano óptimo que separa las diferentes clases en el espacio de características. Se utiliza el kernel RBF (Radial Basis Function) para manejar relaciones no lineales. Este modelo requiere el escalado previo de las variables numéricas, proceso que se realiza automáticamente en el código.

### XGBoost Classifier

XGBoost es un algoritmo de gradient boosting optimizado que ofrece alto rendimiento en problemas de clasificación. Se configura con 100 estimadores, una profundidad máxima de 5 niveles y una tasa de aprendizaje de 0.1. Este modelo es conocido por su velocidad y precisión en conjuntos de datos tabulares.

## Creación de Etiquetas de Productividad

La variable objetivo se crea mediante un proceso de normalización y categorización. Primero, se selecciona el valor de productividad disponible: si existe productividad con plan de negocio, se usa ese valor; de lo contrario, se utiliza la productividad sin plan. Luego, los valores se normalizan según la unidad de medida, convirtiendo todas las unidades a kilogramos por hectárea equivalentes. Finalmente, se aplica la categorización por percentiles: el 40% inferior corresponde a Baja productividad, el rango del 40% al 70% es Media productividad, y el 30% superior es Alta productividad.

## Consideraciones para Producción

El dataset utilizado es estático y contiene solo 6 registros de ejemplo. Para una aplicación en producción real, se recomienda expandir el dataset con datos reales de organizaciones rurales, implementar persistencia de datos mediante una base de datos, y considerar técnicas de validación cruzada para evaluar el rendimiento de los modelos. También sería conveniente implementar un sistema de reentrenamiento periódico con nuevos datos para mantener la precisión del modelo a lo largo del tiempo.

## Licencia

Este proyecto está desarrollado con fines educativos y demostrativos. Feel free de modificar y adaptar el código según tus necesidades específicas.
