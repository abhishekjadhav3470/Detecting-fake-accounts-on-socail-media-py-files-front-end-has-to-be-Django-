"""detecting_fake_account URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from django.conf.urls.static import static
from detecting_fake_account import settings
from socialuser import views as socialuserviews
urlpatterns = [
    path('admin/', admin.site.urls),

    url('^$',socialuserviews.socialuser_index, name="socialuser_index"),
    url(r'^socialuser_login/$',socialuserviews.socialuser_login, name="socialuser_login"),
    url(r'^socialuser_register/$',socialuserviews.socialuser_register, name="socialuser_register"),
    url(r'^socialuser_home/$',socialuserviews.socialuser_home, name="socialuser_home"),
    url(r'^csvdataview/$',socialuserviews.csvdataview, name="csvdataview"),
    url(r'^svm_algorithm/$',socialuserviews.svm_algorithm, name="svm_algorithm"),
    url(r'^random_forest/$',socialuserviews.random_forest, name="random_forest"),
    url(r'^neural_network/$',socialuserviews.neural_network, name="neural_network"),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
