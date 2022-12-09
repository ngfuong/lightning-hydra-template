from django.shortcuts import render
from employee.forms import UserImage
from employee.visual_models import Predictor

model = Predictor()


def image_request(request):
    if request.method == "POST":
        form = UserImage(request.POST, request.FILES)
        if form.is_valid():
            form.save()

            # Getting the current instance object to display in the template
            img_object = form.instance
            model.predict(img_object.image.url)

            read_image = "media/images/query.jpg"
            return render(
                request, "image.form.html", {"form": form, "query_img": img_object.image.url ,"img_obj": read_image}
            )
    else:
        form = UserImage()

    return render(request, "image.form.html", {"form": form})
