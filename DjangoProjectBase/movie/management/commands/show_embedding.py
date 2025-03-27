import os
import numpy as np
import random
from django.core.management.base import BaseCommand
from movie.models import Movie
from openai import OpenAI
from dotenv import load_dotenv

class Command(BaseCommand):
    help = "Selecciona una película al azar y muestra su embedding usando OpenAI"

    def handle(self, *args, **kwargs):
        # ✅ Cargar la API Key de OpenAI
        load_dotenv('../openAI.env')
        client = OpenAI(api_key=os.environ.get('openai_apikey'))

        # ✅ Obtener una película al azar
        movie_count = Movie.objects.count()
        if movie_count == 0:
            self.stdout.write("❌ No hay películas en la base de datos.")
            return

        random_index = random.randint(0, movie_count - 1)
        movie = Movie.objects.all()[random_index]

        self.stdout.write(f"\U0001F3AC Película seleccionada: {movie.title}")

        # ✅ Generar embedding de la descripción
        def get_embedding(text):
            response = client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            return np.array(response.data[0].embedding, dtype=np.float32)

        embedding = get_embedding(movie.description)

        # ✅ Mostrar los primeros valores del embedding para visualización
        self.stdout.write(f"\U0001F4A1 Embedding de '{movie.title}':")
        self.stdout.write(str(embedding[:5]) + " ...")  # Mostramos solo los primeros 10 valores

