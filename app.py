"""
Rural Productivity Classifier - SOLO INTERFAZ
Versión simplificada SIN entrenamiento de modelos
Solo simula predicciones basadas en reglas simples
"""

from flask import Flask, render_template, request

app = Flask(__name__)

# ============================================================================
# LÓGICA SIMPLIFICADA - SOLO PARA DEMO
# ============================================================================

def predecir_productividad_simple(datos):
    """
    Predicción simple basada en reglas (NO es ML real).
    Solo para mostrar la interfaz funcionando.
    """
    # Calcular un "score" simple basado en los datos
    score = 0
    
    # Índice organizacional (peso: 30%)
    score += (datos['indice_org'] / 100) * 30
    
    # Nivel educativo (peso: 20%)
    score += (datos['nivel_educativo'] / 3) * 20
    
    # Tiempo de ejecución (peso: 15%)
    if datos['tiempo_ejecucion'] >= 12:
        score += 15
    elif datos['tiempo_ejecucion'] >= 6:
        score += 10
    else:
        score += 5
    
    # Brecha territorial (peso: 15%)
    if datos['brecha_territorial'] == 'baja':
        score += 15
    elif datos['brecha_territorial'] == 'media':
        score += 10
    else:
        score += 5
    
    # Condiciones climáticas (peso: 20%)
    if datos['sequia'] == 0 and datos['precipitacion'] >= 1000:
        score += 20
    elif datos['sequia'] == 1:
        score += 12
    else:
        score += 5
    
    # Clasificar según el score
    if score >= 70:
        return 'Alta'
    elif score >= 45:
        return 'Media'
    else:
        return 'Baja'

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
        # Obtener datos del formulario
        datos_usuario = {
            'indice_org': float(request.form.get('indice_org', 50)),
            'porcentaje_mujeres': float(request.form.get('pct_mujeres', 50)),
            'porcentaje_varones': float(request.form.get('pct_varones', 50)),
            'nivel_educativo': int(request.form.get('nivel_educativo', 2)),
            'tiempo_ejecucion': int(request.form.get('tiempo_ejecucion', 12)),
            'precipitacion': float(request.form.get('precipitacion', 1000)),
            'temperatura': float(request.form.get('temperatura', 20)),
            'sequia': int(request.form.get('sequia', 0)),
            'producto': request.form.get('tipo_producto', 'Cafe'),
            'brecha_territorial': request.form.get('brecha_territorial', 'media'),
            'modelo_ml': request.form.get('modelo_ml', 'random_forest')
        }
        
        # Hacer "predicción" simple
        resultado = predecir_productividad_simple(datos_usuario)
        
        # Determinar mensaje y estilo
        if resultado == 'Alta':
            clase_css = 'success'
            mensaje = 'La organización tiene ALTA productividad potencial'
        elif resultado == 'Media':
            clase_css = 'warning'
            mensaje = 'La organización tiene MEDIA productividad potencial'
        else:
            clase_css = 'danger'
            mensaje = 'La organización tiene BAJA productividad potencial'
        
        # Formatear datos para mostrar
        datos_display = {
            'indice_desarrollo_organizacional': datos_usuario['indice_org'],
            'nivel_educativo_promedio': datos_usuario['nivel_educativo'],
            'porcentaje_mujeres': datos_usuario['porcentaje_mujeres'],
            'porcentaje_varones': datos_usuario['porcentaje_varones'],
            'producto': datos_usuario['producto'],
            'tiempo_ejecucion_meses': datos_usuario['tiempo_ejecucion'],
            'brecha_territorial': datos_usuario['brecha_territorial'],
            'cambio_climatico_precipitacion': datos_usuario['precipitacion'],
            'cambio_climatico_temperatura': datos_usuario['temperatura'],
            'cambio_climatico_secuela': datos_usuario['sequia']
        }
        
        return render_template(
            'index.html',
            prediccion=resultado,
            clase_css=clase_css,
            mensaje=mensaje,
            datos=datos_display,
            modelo_seleccionado=datos_usuario['modelo_ml']
        )
        
    except Exception as e:
        return render_template(
            'index.html',
            error=f"Error al procesar la predicción: {str(e)}"
        )

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)