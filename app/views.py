from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from django.http import HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import librosa
import numpy as np
import time

# Create your views here.

@csrf_exempt
def index(request):
  template = loader.get_template('index.html')
  if request.method == 'POST':
    file1 = request.FILES['file1']
    file2 = request.FILES['file2']
    fs = FileSystemStorage()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    fs.save(timestr + file1.name, file1)
    fs.save(timestr + file2.name, file2)

    # Load reference audio file
    # static/media/input.mp3
    reference_path = 'static/media/' + timestr + file1.name
    reference_audio, reference_sr = librosa.load(reference_path)

    # Load input audio file
    input_path = 'static/media/' + timestr + file2.name
    input_audio, input_sr = librosa.load(input_path)

    # Perform pitch detection
    reference_pitches, _ = librosa.piptrack(y=reference_audio, sr=reference_sr)
    input_pitches, _ = librosa.piptrack(y=input_audio, sr=input_sr)
    reference_pitches_flat = reference_pitches.flatten()
    input_pitches_flat = input_pitches.flatten()

    minimum = min(reference_pitches_flat.shape[0], input_pitches_flat.shape[0])
    reference_pitches_flat = reference_pitches_flat[:minimum]
    input_pitches_flat = input_pitches_flat[:minimum]

    # Calculate the dot product and norms
    dot_product = np.dot(reference_pitches_flat, input_pitches_flat)
    norm_reference = np.linalg.norm(reference_pitches_flat)
    norm_input = np.linalg.norm(input_pitches_flat)

    # Calculate cosine similarity
    similarity = dot_product / (norm_reference * norm_input)

    # Scale the similarity to a percentage between 0 and 100
    similarity_percentage = (similarity * (10000/3))

    if(similarity_percentage > 100):
      similarity_percentage = 100
    
    if(similarity_percentage < 0):
      similarity_percentage = 0

    output = str(similarity_percentage)[:5]

    return render(request, 'index.html', {
            'similarity': output
    })
  return HttpResponse(template.render())

