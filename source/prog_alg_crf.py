import nltk
from nltk.corpus import treebank
from sklearn.model_selection import train_test_split
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from collections import Counter

# Загрузка данных
nltk.download('treebank')
nltk.download('universal_tagset')

# Загрузка данных с Universal Tagset
sentences = treebank.tagged_sents(tagset='universal')

# Проверка данных
print("Пример предложения:", sentences[0])
print("Уникальные теги:", set(tag for sent in sentences for (word, tag) in sent))

# Разделение на обучающую и тестовую выборки
train_sents, test_sents = train_test_split(sentences, test_size=0.2, random_state=42)


# Извлечение признаков
def extract_features(sentence):
    words = [word for (word, tag) in sentence]
    features = []
    for i, word in enumerate(words):
        feats = {
            'word': word,
            'word.lower()': word.lower(),
            'suffix': word[-3:],
            'isupper': word.isupper(),
            'istitle': word.istitle(),
            'isdigit': word.isdigit(),
            'prev_word': words[i - 1] if i > 0 else '<START>',
            'next_word': words[i + 1] if i < len(words) - 1 else '<END>',
        }
        features.append(feats)
    return features


def get_labels(sentence):
    return [tag for (word, tag) in sentence]


def get_tokens(sentence):
    return [word for (word, tag) in sentence]


# Подготовка данных
X_train = [extract_features(s) for s in train_sents]
y_train = [get_labels(s) for s in train_sents]

X_test = [extract_features(s) for s in test_sents]
y_test = [get_labels(s) for s in test_sents]

# Инициализация и обучение CRF модели
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

# Обучение модели
crf.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred = crf.predict(X_test)

labels = list(crf.classes_)
if 'X' in labels:
    labels.remove('X')

# Удаление пунктуации для оценки
ignore_tags = {'PUNCT'}
filtered_labels = [l for l in labels if l not in ignore_tags]

print("\nClassification Report:")
print(metrics.flat_classification_report(
    y_test, y_pred, labels=filtered_labels, digits=3
))

# Общая точность
y_test_flat = [item for sublist in y_test for item in sublist]
y_pred_flat = [item for sublist in y_pred for item in sublist]

from sklearn.metrics import accuracy_score

print("\nОбщая точность (включая пунктуацию):",
      f"{accuracy_score(y_test_flat, y_pred_flat):.4f}")

# Точность без учета пунктуации
y_test_filtered = [tag for tag in y_test_flat if tag not in ignore_tags]
y_pred_filtered = [y_pred_flat[i] for i, tag in enumerate(y_test_flat)
                   if tag not in ignore_tags]

print("Точность без пунктуации:",
      f"{accuracy_score(y_test_filtered, y_pred_filtered):.4f}")


# Анализ ошибок
def print_errors(sentences, y_true, y_pred, num=5):
    error_count = 0
    for i in range(len(sentences)):
        if error_count >= num:
            break
        for j in range(len(sentences[i])):
            if y_true[i][j] != y_pred[i][j] and y_true[i][j] not in ignore_tags:
                print(f"\nПример ошибки {error_count + 1}:")
                print(f"Предложение: {' '.join(get_tokens(sentences[i]))}")
                print(f"Слово: {sentences[i][j][0]}")
                print(f"Истинный тег: {y_true[i][j]}")
                print(f"Предсказанный тег: {y_pred[i][j]}")
                error_count += 1
                if error_count >= num:
                    break


print("\nАнализ ошибок:")
print_errors(test_sents, y_test, y_pred)





# print("\nРаспределение тегов в y_test:", Counter(y_test_flat))
# print("Распределение тегов в y_pred:", Counter(y_pred_flat))