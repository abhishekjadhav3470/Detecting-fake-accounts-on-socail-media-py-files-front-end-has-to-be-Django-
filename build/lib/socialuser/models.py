from django.db import models

# Create your models here.
class user_reg(models.Model):
    id = models.AutoField(primary_key=True)
    full_name = models.CharField(max_length=300)
    email = models.CharField(max_length=300)
    mobile = models.CharField(max_length=300)
    gender = models.CharField(max_length=300)
    place = models.CharField(max_length=300)
    uname = models.CharField(max_length=300)
    password = models.CharField(max_length=300)