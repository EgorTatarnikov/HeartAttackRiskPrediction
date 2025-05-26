import joblib
import pandas as pd
from ds_lib_tatarnikov import to_snake_case
import numpy as np

# Для обработки данных с пропусками модель будем оборачивать в класс, который предсказывает '0.0', если в данных есть пропуск, и применяет модель для предсказания во всех других случаях.
class PredictZeroIfMissing:
    """
    Обёртка для модели, возвращающая 0, если в данных есть пропуск.
    """
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        X = pd.DataFrame(X)  # На случай, если передали не DataFrame
        mask_missing = X.isnull().any(axis=1)
        
        y_pred = np.zeros(len(X), dtype=int)
        if (~mask_missing).sum() > 0:
            y_pred[~mask_missing] = self.model.predict(X[~mask_missing])
        return y_pred
    
    def predict_proba(self, X):
        X = pd.DataFrame(X)
        mask_missing = X.isnull().any(axis=1)
        
        y_proba = np.zeros((len(X), 2))
        y_proba[mask_missing] = [1.0, 0.0]
        if (~mask_missing).sum() > 0:
            y_proba[~mask_missing] = self.model.predict_proba(X[~mask_missing])
        return y_proba


def load_data(input_file):
    """
    Загружает CSV-файл c данными.
    """
    # Загружаем  данные
    data = pd.read_csv(input_file, index_col=0)
    return data

    
def preprocess_data(data):
    """
    Предобработка данных: переименование столбцов, выбор нужных признаков, преобразование типов.
    """
    # Делаем копию загруженного датасета для предобработки
    data_copy = data.copy()
    # Приводим названия к змеиному стилю
    data_copy.columns = [to_snake_case(col) for col in data_copy.columns]
    # Загружаем список признаков, на которых обучалась модель
    feature_list = joblib.load('best_model_feature_list.pkl')
    # Выбираем признаки для предсказания
    data_copy = data_copy[feature_list + ['id']].copy()
    # Приведем признаки к нужному типу
    cat_features = ['diabetes', 'family_history', 'smoking', 'obesity', 'alcohol_consumption',
               'diet', 'previous_heart_problems', 'medication_use', 'gender']
    for feature in cat_features:
        data_copy[feature] = data_copy[feature].astype('string')
    return data_copy 


def load_model(model_pkl = 'best_model.pkl'):
    """
    Загружает модель из файла и оборачивает её в PredictZeroIfMissing.
    """
    # Загрузка модели
    model = joblib.load(model_pkl)
    # Оборачиваем модель в класс
    wrapped_model = PredictZeroIfMissing(model)
    return wrapped_model


def make_prediction(data, model):
    """
    Делает предсказание и сохраняет результат в файл.
    """
    # Устанавливаем порог
    threshold = 0.105
    # Делаем предсказания
    predictions = model.predict_proba(data.drop('id', axis=1))[:, 1]
    predictions = (predictions >= threshold).astype(int)
    float_predictions = predictions.astype(float)
    # Формируем выходной файл
    output_df = pd.DataFrame({
        'id': data['id'],
        'prediction': float_predictions
    })
    return output_df