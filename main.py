from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import pandas as pd

real_movies = ['Побег из Шоушенка',
 'Тёмный рыцарь',
 'Властелин колец: Возвращение короля',
 'Криминальное чтиво',
 'Список Шиндлера',
 'Достать ножи',
 'Кабинет доктора Каллигари',
 'Мстители',
 'Фаворитка',
 'Убийство священного оленя',
 'Убить Билла',
 'Зеркало',
 'На западном фронте без перемен',
 'Мальчишник в Вегасе',
 'Муви 43']

queries = ['На северном фронте нет изменений', 'Девичник в пекине', 'Обед Була', 'Лучшая']

def find_movie_name(orig, input_words, k=5):
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    embeddings = []
    for title in tqdm(orig):
        embedding = model.encode(title, convert_to_tensor=True)
        embeddings.append(embedding)

    all_predictions = []
    for input_word in tqdm(input_words):
        input_embedding = model.encode(input_word, convert_to_tensor = True)

        cosine_score = util.pytorch_cos_sim(input_embedding, torch.stack(embeddings))[0]

        results = torch.topk(cosine_score, k=k)

        movies = []
        for score, index in zip(results[0], results[1]):
            movies.append(orig[index])

        all_predictions.append(movies)

    return all_predictions

print(find_movie_name(real_movies, queries))