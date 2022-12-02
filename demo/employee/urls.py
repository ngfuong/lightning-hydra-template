from django.urls import path
from employee.views import image_request

app_name = "employee"
urlpatterns = [path("", image_request, name="image-request")]
