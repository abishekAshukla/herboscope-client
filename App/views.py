from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from keras.models import load_model
from PIL import Image
import cv2
import numpy as np
import base64
import time

# Create your views here.
def index(request):
    return render(request, "index.html")

@csrf_exempt
def preprocess(request):
    if request.method == 'POST' and 'file' in request.FILES:

        image_file = request.FILES['file']
        image = Image.open(image_file)
        image = image.resize((200, 200))
        image = np.array(image)
        
        x1 = int(image.shape[0])
        y1 = int(image.shape[1])

        grey_scale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        grey_scale = cv2.resize(grey_scale, (x1, y1))

        _, threshold = cv2.threshold(grey_scale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        grey_image_base64 = base64.b64encode(cv2.imencode('.jpg', grey_scale)[1]).decode('utf-8')
        threshold_image_base64 = base64.b64encode(cv2.imencode('.jpg', threshold)[1]).decode('utf-8')

        response_data = {
            'grey_image': grey_image_base64,
            'threshold_image': threshold_image_base64
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'No file provided'}, status=400)

@csrf_exempt
def prediction(request):
    if request.method == 'POST' and 'file' in request.FILES:

        path = 'D:\\Aadi\\Projects\\AI_ML_DL\\Herbs_Identification\\Model\\inception_v3_model.h5'
        model = load_model(path,compile=False)

        start = time.time()

        image_file = request.FILES['file']
        image = Image.open(image_file)
        image = image.resize((75, 75))
        image = np.array(image)
        image = image.reshape(1, 75, 75, 3)
        image = image.astype('float32')
        image = image / 255.0

        prediction = model.predict(image)
        predicted_herb = np.argmax(prediction)
        predicted_herb = str(predicted_herb)

        end = time.time()
        execution_time = "Execution Time: {0:.4} seconds".format(end-start)

        response_data = {
            'predicted_herb': predicted_herb,
            'execution_time': execution_time
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'No file provided'}, status=400)
    