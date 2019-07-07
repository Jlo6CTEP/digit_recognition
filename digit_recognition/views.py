import random

import numpy
from django.http import HttpResponse
from django.shortcuts import render, render_to_response

# Create your views here.
from django.template import RequestContext

from neural_network.digit_guess import guess_digit

responses = [
    'Yer digit is {}',
    'here is my guess: {}',
    'Im flawless, it is {}',
    'Is this {}?',
    'Here is yer digit {}',
    'It is {}, why not? ',
    'It is {} am i right?',
    '{} is yer digit'
]


def home(request):
    return render(request, 'digit_recognition/home.html')


def upload(request):
    if request.method == 'POST':
        data = request.FILES['img'].read()
        dim = int(numpy.sqrt(len(data) / 4))
        img = numpy.frombuffer(data, dtype=numpy.uint8)[3::4].reshape((dim, dim))
        digit = guess_digit(img)
        if digit is not None:
            return HttpResponse(random.sample(responses, 1)[0].format(digit))
        return HttpResponse('Digit is too small')
    return HttpResponse('error')
