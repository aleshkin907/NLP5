import nltk
from nltk.corpus import brown
from collections import defaultdict, Counter
import numpy as np

# Загрузка данных
nltk.download('brown')
nltk.download('universal_tagset')

tagged_sentences = brown.tagged_sents(tagset='universal')

# Разделение на обучающую и тестовую выборки
split = int(0.8 * len(tagged_sentences))
train_data = tagged_sentences[:split]
test_data = tagged_sentences[split:]

# Подсчет частот слов и тегов
word_counts = Counter()
tag_counts = Counter()
for sentence in train_data:
    for word, tag in sentence:
        word_counts[word] += 1
        tag_counts[tag] += 1

# Определение редких слов (менее 2 вхождений)
rare_words = {word for word, count in word_counts.items() if count < 2}

# Начальные вероятности
pi = defaultdict(float)
# Подсчёт переходов между тегами
transitions = defaultdict(lambda: defaultdict(int))
# Матрица эмиссии
emissions = defaultdict(lambda: defaultdict(int))

# Подсчет частот
for sentence in train_data:
    # Обработка первого слова для начальных вероятностей
    first_tag = sentence[0][1]
    pi[first_tag] += 1

    # Подсчет переходов между тегами
    for i in range(1, len(sentence)):
        prev_tag = sentence[i - 1][1]
        curr_tag = sentence[i][1]
        transitions[prev_tag][curr_tag] += 1

    # Подсчет эмиссий (слово -> тег)
    for word, tag in sentence:
        processed_word = '<UNK>' if word in rare_words else word.lower()
        emissions[tag][processed_word] += 1

# Нормализация начальных вероятностей
total_pi = sum(pi.values())
for tag in pi:
    pi[tag] /= total_pi

# Нормализация матрицы переходов
for prev_tag in transitions:
    total = sum(transitions[prev_tag].values())
    for curr_tag in transitions[prev_tag]:
        transitions[prev_tag][curr_tag] /= total

# Нормализация матрицы эмиссии
for tag in emissions:
    total = sum(emissions[tag].values())
    for word in emissions[tag]:
        emissions[tag][word] /= total

# Получаем список всех тегов
tags = list(tag_counts.keys())


def viterbi(words, tags, pi, transitions, emissions):
    # Инициализация матриц вероятностей и обратных указателей
    V = [{}]
    path = {}

    # Первый шаг
    for tag in tags:
        word = words[0].lower()
        word = word if word in emissions[tag] else '<UNK>'
        V[0][tag] = pi[tag] * emissions[tag].get(word, 1e-10)
        path[tag] = [tag]

    # Рекурсивный шаг
    for t in range(1, len(words)):
        V.append({})
        new_path = {}

        for tag in tags:
            max_prob, best_prev = max(
                (V[t - 1][prev_tag] * transitions[prev_tag].get(tag, 1e-10) * emissions[tag].get(words[t].lower(),
                                                                                                 1e-10), prev_tag)
                for prev_tag in tags
            )
            V[t][tag] = max_prob
            new_path[tag] = path[best_prev] + [tag]

        path = new_path

    # Выбор наилучшего пути
    best_tag = max(V[-1], key=V[-1].get)
    return path[best_tag]


def evaluate(test_data, tags, pi, transitions, emissions):
    correct = 0
    total = 0

    for sentence in test_data:
        words = [word for word, tag in sentence]
        true_tags = [tag for word, tag in sentence]

        if len(words) == 0:
            continue

        predicted_tags = viterbi(words, tags, pi, transitions, emissions)

        for true, pred in zip(true_tags, predicted_tags):
            if true == pred:
                correct += 1
            total += 1

    return correct / total


# Оценка HMM
hmm_accuracy = evaluate(test_data, tags, pi, transitions, emissions)
print(f"HMM Accuracy: {hmm_accuracy:.4f}")


# Функция для оценки без замены редких слов
def evaluate_without_unk(test_data, tags, pi, transitions, emissions_without_unk):
    correct = 0
    total = 0

    for sentence in test_data:
        words = [word for word, tag in sentence]
        true_tags = [tag for word, tag in sentence]

        if len(words) == 0:
            continue

        # Модифицированная версия Viterbi без замены на UNK
        V = [{}]
        path = {}

        for tag in tags:
            word = words[0].lower()
            V[0][tag] = pi[tag] * emissions_without_unk[tag].get(word, 1e-10)
            path[tag] = [tag]

        for t in range(1, len(words)):
            V.append({})
            new_path = {}

            for tag in tags:
                max_prob, best_prev = max(
                    (V[t - 1][prev_tag] * transitions[prev_tag].get(tag, 1e-10) * emissions_without_unk[tag].get(
                        words[t].lower(), 1e-10), prev_tag)
                    for prev_tag in tags
                )
                V[t][tag] = max_prob
                new_path[tag] = path[best_prev] + [tag]

            path = new_path

        predicted_tags = path[max(V[-1], key=V[-1].get)]

        for true, pred in zip(true_tags, predicted_tags):
            if true == pred:
                correct += 1
            total += 1

    return correct / total


# Создаем матрицу эмиссии без замены на UNK
emissions_without_unk = defaultdict(lambda: defaultdict(float))
for sentence in train_data:
    for word, tag in sentence:
        emissions_without_unk[tag][word.lower()] += 1

for tag in emissions_without_unk:
    total = sum(emissions_without_unk[tag].values())
    for word in emissions_without_unk[tag]:
        emissions_without_unk[tag][word] /= total

# Оценка
accuracy_without_unk = evaluate_without_unk(test_data, tags, pi, transitions, emissions_without_unk)
print(f"Accuracy without UNK: {accuracy_without_unk:.4f}")
print(f"Accuracy with UNK: {hmm_accuracy:.4f}")