"""
Rural Productivity Classifier - Aplicación de Machine Learning
Clasifica organizaciones rurales según su nivel de productividad
Versión optimizada para Render - SIN XGBoost
"""

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================================================================
# DATASET ESTÁTICO - Organizaciones Rurales
# ============================================================================

def get_dataset_estatico():
    """Dataset simulado con 6 registros de organizaciones rurales."""
    data = {
        'promedio_productividad_sp': [1200, 850, 2100, 1800, 650, 2500],
        'promedio_productividad_cp': [1500, 1100, 2300, 2100, 900, 2800],
        'unidad_medida': ['kg/ha', 'kg/ha', 'qq/ha', 'qq/ha', 'lt/vaca', 'kg/ha'],
        'indice_desarrollo_organizacional': [85, 62, 92, 78, 55, 95],
        'porcentaje_mujeres': [35, 45, 28, 40, 50, 30],
        'porcentaje_varones': [65, 55, 72, 60, 50, 70],
        'nivel_educativo_promedio': [3, 2, 3, 2, 1, 3],
        'tipo_producto': ['Cafe', 'Leche', 'Palta', 'Cafe', 'Leche', 'Palta'],
        'tiempo_ejecucion_meses': [12, 18, 24, 15, 8, 20],
        'brecha_territorial': ['baja', 'media', 'alta', 'baja', 'media', 'alta'],
        'cambio_climatico_precipitacion': [1200, 950, 1500, 1100, 800, 1600],
        'cambio_climatico_temperatura': [18, 22, 15, 19, 24, 16],
        'cambio_climatico_secuela': [0, 1, 2, 1, 2, 0]
    }
    return pd.DataFrame(data)

# ============================================================================
# CREACIÓN DE ETIQUETAS
# ============================================================================

def crear_etiqueta_productividad(df):
    """Crea la variable objetivo a partir de los valores de productividad."""
    df = df.copy()
    
    df['productividad_base'] = df.apply(
        lambda row: row['promedio_productividad_cp'] 
        if pd.notna(row['promedio_productividad_cp']) and row['promedio_productividad_cp'] > 0 
        else row['promedio_productividad_sp'], 
        axis=1
    )
    
    conversion_factors = {
        'kg/ha': 1.0,
        'qq/ha': 46.0,
        'lt/vaca': 0.125
    }
    
    df['productividad_normalizada'] = df.apply(
        lambda row: row['productividad_base'] * conversion_factors.get(row['unidad_medida'], 1.0),
        axis=1
    )
    
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
# PREPROCESAMIENTO
# ============================================================================

def preprocesar_datos(df, is_training=True):
    """Preprocesa los datos para el modelo."""
    df = df.copy()
    
    productos = ['Cafe', 'Leche', 'Palta']
    for producto in productos:
        df[f'producto_{producto}'] = (df['tipo_producto'] == producto).astype(int)
    
    brecha_map = {'baja': 1, 'media': 2, 'alta': 3}
    df['brecha_codificada'] = df['brecha_territorial'].map(brecha_map)
    
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
# MODELOS
# ============================================================================

def entrenar_modelo(modelo_tipo, X_train, y_train):
    """Entrena el modelo seleccionado."""
    scaler = None
    necesita_escalado = False
    
    if modelo_tipo == 'random_forest':
        modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
        modelo.fit(X_train, y_train)
        
    elif modelo_tipo == 'svm':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        modelo = SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        modelo.fit(X_train_scaled, y_train)
        necesita_escalado = True
        
    else:  # Fallback a Random Forest
        modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
        modelo.fit(X_train, y_train)
    
    return modelo, scaler, necesita_escalado

def predecir(modelo, X_pred, scaler=None, necesita_escalado=False):
    """Realiza la predicción."""
    if necesita_escalado and scaler is not None:
        X_pred_scaled = scaler.transform(X_pred)
        return modelo.predict(X_pred_scaled)
    else:
        return modelo.predict(X_pred)

# ============================================================================
# RUTAS
# ============================================================================

@app.route('/')
def index():
    """Renderiza la página principal."""
    return render_template('index.html', datos=None)

@app.route('/predict', methods=['POST'])
def predict():
    """Procesa la solicitud de predicción."""
    try:
        modelo_tipo = request.form.get('modelo_ml', 'random_forest')
        
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
        
        # Entrenar
        df_entrenamiento = get_dataset_estatico()
        df_entrenamiento = crear_etiqueta_productividad(df_entrenamiento)
        X_train, y_train = preprocesar_datos(df_entrenamiento, is_training=True)
        
        modelo, scaler, necesita_escalado = entrenar_modelo(modelo_tipo, X_train, y_train)
        
        # Predecir
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
        prediccion = predecir(modelo, X_pred, scaler, necesita_escalado)
        resultado = prediccion[0]
        
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
    app.run(debug=False, host='0.0.0.0', port=5000)