from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django_backend_app.prediction import Prediction


class PredNlpModel(APIView):
    def post(self, request):
        prediction = Prediction()
        response = prediction.predict(request)

        if isinstance(response, HttpResponse):
            return response

        return Response(response["prediction"], status=response["status"])

    def get(self, request):
        prediction = Prediction()
        response = prediction.get_subtitle(request)

        if isinstance(response, HttpResponse):
            return response

        return Response(response["subtitle"], status=response["status"])
