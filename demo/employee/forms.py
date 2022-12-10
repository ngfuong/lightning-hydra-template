from django import forms
from employee.models import UploadImage


class UserImage(forms.ModelForm):
    class Meta:
        # To specify the model to be used to create form
        model = UploadImage
        # It includes all the fields of model
        fields = "__all__"
