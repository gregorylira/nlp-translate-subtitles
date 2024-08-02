from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django_backend_app.prediction import Prediction

class PredNlpModel(APIView):
    def post(self, request):
        prediction = Prediction()
        return_dict = prediction.predict(request)
        return Response(return_dict["prediction"], status=return_dict["status"])