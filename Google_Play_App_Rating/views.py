from django.shortcuts import render, HttpResponse
import pickle
import numpy as np
import pandas as pd

with open("googleplayrating.pkl", "rb")as file:
    model = pickle.load(file)


# Create your views here.
def home(request):
    res = 0
    try:
        if request.method == "POST":
            name = request.POST["name"]
            category = request.POST["category"]
            reviews = request.POST["reviews"]
            size = request.POST["size"]
            installs = request.POST["installs"]
            type = request.POST["type"]
            price = request.POST["price"]
            content = request.POST["content"]
            version = request.POST["version"]
            year = request.POST["year"]
            month = request.POST["month"]
            day = request.POST["day"]

            data = np.array([int(category), float(reviews), float(size), int(installs), int(type), float(price), int(content), float(version), int(year), int(month), int(day)])
            data = data.reshape(1, -1)
            print(data)
        
            res = model.predict(data).round(1)[0]

            return render(request, "output.html", {'response': res})

    except:
        return render(request, 'index.html', {'error': 'Please fill all value'})
        
    return render(request, "index.html")
