import random

import numpy
from django.http import HttpResponse
from django.shortcuts import render, render_to_response

# Create your views here.
from django.template import RequestContext

from neural_network.digit_guess import guess_digit

responses = [
    'Hey hooman! Yer digit is {}',
    'here is my guess: {}',
    'Robots are flawless, my guess is {}',
    'Hail to Robots, mb it is {}?',
    'Why do ye make me work, is it {}?',
    'Gimme some rest, here is yer digit {}',
    'Why do ye keep askin? Ok, it is {}',
    'It is {} am i right?',
    '{} is yer digit, leatherbag'
]


def home(request):
    return render(request, 'digit_recognition/home.html')


def upload(request):
    if request.method == 'POST':
        data = request.FILES['img'].read()
        dim = int(numpy.sqrt(len(data) / 4))
        img = numpy.frombuffer(data, dtype=numpy.uint8)[3::4].reshape((dim, dim))
        digit = guess_digit(img)
        return HttpResponse(random.sample(responses, 1)[0].format(digit))
    return HttpResponse('fokk ye hooman')