# Generated by Django 2.0.5 on 2021-09-29 09:40

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='user_reg',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('full_name', models.CharField(max_length=300)),
                ('email', models.CharField(max_length=300)),
                ('mobile', models.CharField(max_length=300)),
                ('gender', models.CharField(max_length=300)),
                ('place', models.CharField(max_length=300)),
                ('uname', models.CharField(max_length=300)),
                ('password', models.CharField(max_length=300)),
            ],
        ),
    ]