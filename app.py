"""
Rural Productivity Classifier - SOLO INTERFAZ VISUAL
Sin procesamiento de modelos ML - Solo formulario de presentación
"""

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    """Muestra el formulario principal."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recibe los datos y muestra un mensaje simple."""
    try:
        # Obtener datos del formulario
        datos = {
            'modelo': request.form.get('modelo_ml', 'Random Forest'),
            'indice_org': request.form.get('indice_org', '50'),
            'nivel_educativo': request.form.get('nivel_educativo', '2'),
            'pct_mujeres': request.form.get('pct_mujeres', '50'),
            'pct_varones': request.form.get('pct_varones', '50'),
            'producto': request.form.get('tipo_producto', 'Cafe'),
            'tiempo_ejecucion': request.form.get('tiempo_ejecucion', '12'),
            'brecha_territorial': request.form.get('brecha_territorial', 'media'),
            'precipitacion': request.form.get('precipitacion', '1000'),
            'temperatura': request.form.get('temperatura', '20'),
            'sequia': request.form.get('sequia', '0')
        }
        
        # Mostrar datos recibidos
        return render_template('index.html', datos=datos, mensaje="Datos recibidos correctamente")
        
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)