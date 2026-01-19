"""
Rural Productivity Classifier - Aplicación de Machine Learning
Clasifica organizaciones rurales según su nivel de productividad
"""

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================================================================
# DATASET ESTÁTICO - Organizaciones Rurales
# ============================================================================

def get_dataset_estatico():
    """
    Dataset simulado con 6 registros de organizaciones rurales.
    Variables incluyen: productividad, variables internas y externas.
    """
    data = {
        # PRODUCTIVIDAD (para crear la etiqueta)
        'promedio_productividad_sp': [1200, 850, 2100, 1800, 650, 2500],
        'promedio_productividad_cp': [1500, 1100, 2300, 2100, 900, 2800],
        'unidad_medida': ['kg/ha', 'kg/ha', 'qq/ha', 'qq/ha', 'lt/vaca', 'kg/ha'],
        
        # VARIABLES INTERNAS
        'indice_desarrollo_organizacional': [85, 62, 92, 78, 55, 95],
        'porcentaje_mujeres': [35, 45, 28, 40, 50, 30],
        'porcentaje_varones': [65, 55, 72, 60, 50, 70],
        'nivel_educativo_promedio': [3, 2, 3, 2, 1, 3],  # 1=Primaria, 2=Secundaria, 3=Superior
        'tipo_producto': ['Cafe', 'Leche', 'Palta', 'Cafe', 'Leche', 'Palta'],
        
        # VARIABLES EXTERNAS
        'tiempo_ejecucion_meses': [12, 18, 24, 15, 8, 20],
        'brecha_territorial': ['baja', 'media', 'alta', 'baja', 'media', 'alta'],
        'cambio_climatico_precipitacion': [1200, 950, 1500, 1100, 800, 1600],
        'cambio_climatico_temperatura': [18, 22, 15, 19, 24, 16],
        'cambio_climatico_secuela': [0, 1, 2, 1, 2, 0]  # 0=Ninguna, 1=Moderada, 2=Severa
    }
    
    return pd.DataFrame(data)


# ============================================================================
# CREACIÓN DE ETIQUETAS (LABEL ENGINEERING)
# ============================================================================

def crear_etiqueta_productividad(df):
    """
    Crea la variable objetivo (label) a partir de los valores de productividad.
    La categorización se realiza mediante percentiles definidos en el código.
    
    Returns:
        df con columna 'productividad_nivel' añadida
    """
    df = df.copy()
    
    # Calcular valor de productividad a usar:
    # Si existe productividad con plan (CP), usarla; si no, usar sin plan (SP)
    df['productividad_base'] = df.apply(
        lambda row: row['promedio_productividad_cp'] 
        if pd.notna(row['promedio_productividad_cp']) and row['promedio_productividad_cp'] > 0 
        else row['promedio_productividad_sp'], 
        axis=1
    )
    
    # Normalización simple por unidad de medida (convertir todo a kg/ha equivalente)
    # Factores de conversión aproximados: 1 qq/ha = 46 kg/ha
    conversion_factors = {
        'kg/ha': 1.0,
        'qq/ha': 46.0,
        'lt/vaca': 1.0 / 8.0  # Aproximado: 8 litros = 1 kg de equivalente
    }
    
    df['productividad_normalizada'] = df.apply(
        lambda row: row['productividad_base'] * conversion_factors.get(row['unidad_medida'], 1.0),
        axis=1
    )
    
    # Categorización mediante percentiles
    # Baja: percentil 0-40, Media: percentil 40-70, Alta: percentil 70-100
    p40 = df['productividad_normalizada'].quantile(0.40)
    p70 = df['productividad_normalizada'].quantile(0.70)
    
    def categorizar(valor):
        if valor <= p40:
            return 'Baja'
        elif valor <= p70:
            return 'Media'
        else:
            return 'Alta'
    
    df['productividad_nivel'] = df['productividad_normalizada'].apply(categorizar)
    
    return df


# ============================================================================
# PREPROCESAMIENTO DE DATOS
# ============================================================================

def preprocesar_datos(df, is_training=True):
    """
    Preprocesa los datos para el modelo de ML.
    Incluye codificación de variables categóricas.
    """
    df = df.copy()
    
    # Codificar variables categóricas
    # Tipo de producto (One-Hot Encoding)
    productos = ['Cafe', 'Leche', 'Palta']
    for producto in productos:
        df[f'producto_{producto}'] = (df['tipo_producto'] == producto).astype(int)
    
    # Brecha territorial (Ordinal Encoding)
    brecha_map = {'baja': 1, 'media': 2, 'alta': 3}
    df['brecha_codificada'] = df['brecha_territorial'].map(brecha_map)
    
    # Seleccionar características para el modelo
    features = [
        'indice_desarrollo_organizacional',
        'porcentaje_mujeres',
        'porcentaje_varones',
        'nivel_educativo_promedio',
        'tiempo_ejecucion_meses',
        'cambio_climatico_precipitacion',
        'cambio_climatico_temperatura',
        'cambio_climatico_secuela',
        'producto_Cafe',
        'producto_Leche',
        'producto_Palta',
        'brecha_codificada'
    ]
    
    X = df[features]
    
    if is_training:
        y = df['productividad_nivel']
        return X, y
    else:
        return X


# ============================================================================
# MODELOS DE MACHINE LEARNING
# ============================================================================

def entrenar_modelo(modelo_tipo, X_train, y_train):
    """
    Entrena el modelo seleccionado y lo retorna.
    """
    if modelo_tipo == 'random_forest':
        modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
    elif modelo_tipo == 'svm':
        # SVM requiere escalado de datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        modelo = SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        # Entrenar SVM con datos escalados
        modelo.fit(X_train_scaled, y_train)
        return modelo, scaler, True  # True indica que necesita escalado
    elif modelo_tipo == 'xgboost':
        modelo = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
    else:
        raise ValueError(f"Modelo no reconocido: {modelo_tipo}")
    
    return modelo, None, False  # None para scaler, False indica no necesita escalado


def predecir(modelo, X_pred, scaler=None, necesita_escalado=False):
    """
    Realiza la predicción con el modelo entrenado.
    """
    if necesita_escalado and scaler is not None:
        X_pred_scaled = scaler.transform(X_pred)
        return modelo.predict(X_pred_scaled)
    else:
        return modelo.predict(X_pred)


# ============================================================================
# RUTAS DE FLASK
# ============================================================================

@app.route('/')
def index():
    """Renderiza la página principal con el formulario."""
    return render_template('index.html', datos=None)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Procesa la solicitud de predicción.
    Recibe datos del formulario, entrena el modelo y retorna el resultado.
    """
    try:
        # Obtener tipo de modelo seleccionado
        modelo_tipo = request.form.get('modelo_ml', 'random_forest')
        
        # Obtener datos del formulario
        datos_usuario = {
            'indice_desarrollo_organizacional': float(request.form.get('indice_org', 50)),
            'porcentaje_mujeres': float(request.form.get('pct_mujeres', 50)),
            'porcentaje_varones': float(request.form.get('pct_varones', 50)),
            'nivel_educativo_promedio': int(request.form.get('nivel_educativo', 2)),
            'tiempo_ejecucion_meses': int(request.form.get('tiempo_ejecucion', 12)),
            'cambio_climatico_precipitacion': float(request.form.get('precipitacion', 1000)),
            'cambio_climatico_temperatura': float(request.form.get('temperatura', 20)),
            'cambio_climatico_secuela': int(request.form.get('sequia', 0)),
            'producto': request.form.get('tipo_producto', 'Cafe'),
            'brecha_territorial': request.form.get('brecha_territorial', 'media')
        }
        
        # Cargar y preparar dataset de entrenamiento
        df_entrenamiento = get_dataset_estatico()
        df_entrenamiento = crear_etiqueta_productividad(df_entrenamiento)
        X_train, y_train = preprocesar_datos(df_entrenamiento, is_training=True)
        
        # Entrenar el modelo
        modelo, scaler, necesita_escalado = entrenar_modelo(modelo_tipo, X_train, y_train)
        modelo.fit(X_train, y_train)
        
        # Preparar datos del usuario para predicción
        df_usuario = pd.DataFrame([{
            'indice_desarrollo_organizacional': datos_usuario['indice_desarrollo_organizacional'],
            'porcentaje_mujeres': datos_usuario['porcentaje_mujeres'],
            'porcentaje_varones': datos_usuario['porcentaje_varones'],
            'nivel_educativo_promedio': datos_usuario['nivel_educativo_promedio'],
            'tiempo_ejecucion_meses': datos_usuario['tiempo_ejecucion_meses'],
            'cambio_climatico_precipitacion': datos_usuario['cambio_climatico_precipitacion'],
            'cambio_climatico_temperatura': datos_usuario['cambio_climatico_temperatura'],
            'cambio_climatico_secuela': datos_usuario['cambio_climatico_secuela'],
            'tipo_producto': datos_usuario['producto'],
            'brecha_territorial': datos_usuario['brecha_territorial']
        }])
        
        X_pred = preprocesar_datos(df_usuario, is_training=False)
        
        # Realizar predicción
        prediccion = predecir(modelo, X_pred, scaler, necesita_escalado)
        resultado = prediccion[0]
        
        # Determinar clase CSS y mensaje según el resultado
        if resultado == 'Alta':
            clase_css = 'success'
            mensaje = 'La organización tiene ALTA productividad potencial'
        elif resultado == 'Media':
            clase_css = 'warning'
            mensaje = 'La organización tiene MEDIA productividad potencial'
        else:
            clase_css = 'danger'
            mensaje = 'La organización tiene BAJA productividad potencial'
        
        return render_template(
            'index.html',
            prediccion=resultado,
            clase_css=clase_css,
            mensaje=mensaje,
            datos=datos_usuario,
            modelo_seleccionado=modelo_tipo
        )
        
    except Exception as e:
        return render_template(
            'index.html',
            error=f"Error al procesar la predicción: {str(e)}"
        )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
