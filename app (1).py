import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import io
import base64

app = Flask(__name__)

# Configurações
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Criar pasta de uploads se não existir
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_array):
    """Converter imagem numpy para base64"""
    _, buffer = cv2.imencode('.png', image_array)
    img_base64 = base64.b64encode(buffer).decode()
    return f"data:image/png;base64,{img_base64}"


def process_image(image_path, threshold1=50, threshold2=150, blur_size=5, iterations=1):
    """
    Processa a imagem para detecção de bordas otimizada.
    - Converte para cinza
    - Aplica desfoque para reduzir ruído de alta frequência
    - Canny para bordas
    - Dilatação/Erosão opcional para fechar gaps
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Não foi possível ler a imagem")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Suavização inicial para remover ruído de pixels isolados
    if blur_size > 0:
        if blur_size % 2 == 0: blur_size += 1
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    # Canny para detecção de bordas
    edges = cv2.Canny(gray, threshold1, threshold2)
    
    # Operações morfológicas para unir bordas próximas e remover pequenos pontos
    if iterations > 0:
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=iterations)
        edges = cv2.erode(edges, kernel, iterations=iterations)
        
    return edges, img


def edges_to_svg(edges_image, min_area=20, epsilon_factor=0.005):
    """
    Converte imagem de bordas para SVG otimizado para corte a laser.
    - Filtra contornos por área (remove ruído)
    - Suaviza contornos com aproximação poligonal controlada
    - Gera path SVG limpo
    """
    height, width = edges_image.shape
    
    # Encontrar contornos com hierarquia para manter furos (RETR_TREE)
    # CHAIN_APPROX_TC89_L1 ou KCOS pode ser usado para curvas, mas SIMPLE + approxPolyDP é mais controlável
    contours, hierarchy = cv2.findContours(edges_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<!-- Gerado por Otimizador de Corte a Laser -->',
        '<style>path { stroke: black; fill: none; stroke-width: 1; vector-effect: non-scaling-stroke; }</style>'
    ]
    
    if contours:
        for i, contour in enumerate(contours):
            # 1. Remover ruído por área mínima
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # 2. Suavização e Otimização de Pontos (Douglas-Peucker)
            # epsilon_factor controla a fidelidade vs quantidade de pontos
            # 0.001 = muito fiel, muitos pontos | 0.02 = muito simplificado
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(simplified) < 2:
                continue
                
            # 3. Gerar Path Data
            path_parts = []
            for j, point in enumerate(simplified):
                x, y = point[0]
                cmd = "M" if j == 0 else "L"
                path_parts.append(f"{cmd}{x} {y}")
            
            path_data = " ".join(path_parts) + " Z"
            svg_lines.append(f'  <path d="{path_data}" />')
    
    svg_lines.append('</svg>')
    return '\n'.join(svg_lines)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Arquivo não selecionado'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Tipo de arquivo não permitido'}), 400
        
        filename = secure_filename(file.filename)
        timestamp = str(int(__import__('time').time() * 1000))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img = cv2.imread(filepath)
        preview_base64 = image_to_base64(img)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'preview': preview_base64
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect-edges', methods=['POST'])
def detect_edges_endpoint():
    try:
        data = request.get_json()
        filename = data.get('filename')
        t1 = int(data.get('threshold1', 50))
        t2 = int(data.get('threshold2', 150))
        blur = int(data.get('blur', 5))
        morph = int(data.get('morph', 1))
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Arquivo não encontrado'}), 404
        
        edges, _ = process_image(filepath, t1, t2, blur, morph)
        edges_base64 = image_to_base64(edges)
        
        return jsonify({
            'success': True,
            'edges': edges_base64,
            'width': edges.shape[1],
            'height': edges.shape[0]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/convert-svg', methods=['POST'])
def convert_svg_endpoint():
    try:
        data = request.get_json()
        filename = data.get('filename')
        t1 = int(data.get('threshold1', 50))
        t2 = int(data.get('threshold2', 150))
        blur = int(data.get('blur', 5))
        morph = int(data.get('morph', 1))
        min_area = float(data.get('min_area', 20))
        epsilon = float(data.get('epsilon', 0.005))
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        edges, _ = process_image(filepath, t1, t2, blur, morph)
        
        svg_content = edges_to_svg(edges, min_area, epsilon)
        
        return jsonify({
            'success': True,
            'svg': svg_content
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download-svg', methods=['POST'])
def download_svg():
    try:
        data = request.get_json()
        svg_content = data.get('svg')
        if not svg_content:
            return jsonify({'error': 'SVG não fornecido'}), 400
        
        return send_file(
            io.BytesIO(svg_content.encode('utf-8')),
            mimetype='image/svg+xml',
            as_attachment=True,
            download_name='laser_cut_ready.svg'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    try:
        data = request.get_json()
        filename = data.get('filename')
        if filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(filepath):
                os.remove(filepath)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
