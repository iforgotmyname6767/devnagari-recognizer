import base64
import io
import json

import numpy as np
from PIL import Image
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

try:
    from ai_edge_litert.interpreter import Interpreter   # small runtime, used on Render
except ImportError:
    import tensorflow as tf                                # fallback on your laptop
    Interpreter = tf.lite.Interpreter

MODEL_DIR = settings.BASE_DIR / 'model'
INTERP = Interpreter(model_path=str(MODEL_DIR / 'devanagari_cnn.tflite'))
INTERP.allocate_tensors()
IN_INDEX = INTERP.get_input_details()[0]['index']
OUT_INDEX = INTERP.get_output_details()[0]['index']
with open(MODEL_DIR / 'labels.json', encoding='utf-8') as f:
    LABELS = json.load(f)


def preprocess(img):
    a = np.array(img.convert('L'))
    ys, xs = np.where(a > 40)                      # find the white strokes
    if len(xs) == 0:
        return None
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]   # crop to the strokes
    h, w = a.shape
    s = max(h, w)
    square = np.zeros((s, s), dtype=np.uint8)      # pad to a square, keep the aspect ratio
    square[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    small = np.array(Image.fromarray(square).resize((28, 28), Image.LANCZOS)) / 255.0
    out = np.zeros((32, 32), dtype='float32')      # 2 px border, like the dataset
    out[2:30, 2:30] = small
    return out.reshape(1, 32, 32, 1)


def run_model(x):
    INTERP.set_tensor(IN_INDEX, x)
    INTERP.invoke()
    return INTERP.get_tensor(OUT_INDEX)[0]


def index(request):
    return render(request, 'recognizer/index.html')


@require_POST
def predict(request):
    data = json.loads(request.body)
    raw = base64.b64decode(data['image'].split(',', 1)[1])
    x = preprocess(Image.open(io.BytesIO(raw)))
    if x is None:
        return JsonResponse({'error': 'Please draw a character first.'}, status=400)
    probs = run_model(x)
    top = probs.argsort()[::-1][:3]
    return JsonResponse({'predictions': [
        {'char': LABELS[i], 'confidence': round(float(probs[i]) * 100, 1)} for i in top]})