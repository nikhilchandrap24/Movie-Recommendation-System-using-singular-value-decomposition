# Generated by Django 4.1.7 on 2023-04-03 17:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('user_interface', '0002_alter_movie_release_date_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='movie',
            name='release_date',
            field=models.CharField(default=None, max_length=150),
        ),
        migrations.AlterField(
            model_name='movie',
            name='video_release_date',
            field=models.CharField(default=None, max_length=150),
        ),
    ]