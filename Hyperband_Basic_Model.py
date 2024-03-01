import os
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import tensorflow as tf
import pandas as pd
from kerastuner.tuners import Hyperband
from sklearn.model_selection import train_test_split

# Wczytanie pliku metadanych
metadata_path = 'data/HAM10000_metadata'
df = pd.read_csv(metadata_path)

# Ścieżki do folderów z obrazami
folder_path = 'data/combined_data'

# Ekstrakcja klas
df = df[df['dx'] != 'df']
df = df[df['dx'] != 'vasc']
df = df[df['dx'] != 'akiec']
df = df[df['dx'] != 'bkl']
df = df.groupby('dx').apply(lambda x: x.sample(n=1000, random_state=42) if x.name == 'nv' else x)
unique_labels = sorted(df['dx'].unique())

# Poprawa sciezek obrazow
df['image_id'] = df['image_id'].apply(lambda x: os.path.join(folder_path, x + '.jpg'))

# Generator walidacyjny i testowy
augmentation_generator = ImageDataGenerator(rescale=1. / 255)
validation_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Podział danych na zbiory treningowy, walidacyjny i testowy
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['dx'])
validation_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['dx'])

# Zliczenie zdjęć w każdej kategorii
counts = train_df['dx'].value_counts()

# Utworzenie generatorów danych
train_generator = augmentation_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_id',
    y_col='dx',
    batch_size=64,
    seed=42,
    shuffle=True,
    class_mode='sparse',
    target_size=(224, 224),
    classes=unique_labels
)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_df,
    x_col='image_id',
    y_col='dx',
    batch_size=64,
    seed=42,
    shuffle=True,
    class_mode='sparse',
    target_size=(224, 224),
    classes=unique_labels
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image_id',
    y_col='dx',
    batch_size=256,
    seed=42,
    shuffle=False,
    class_mode='sparse',
    target_size=(224, 224),
    classes=unique_labels
)

# Rozmiary obrazów
img_height, img_width = 224, 224

# Liczba klas
num_classes = 3

# Model do algorytmu Hyperband
def build_model(hp):
    model = keras.Sequential()

    # Pierwsza warstwa konwolucyjna
    model.add(keras.layers.Conv2D(filters=hp.Int('conv_first_filters', min_value=32, max_value=64, step=32),  # 2
                                  kernel_size=3,
                                  activation='relu',
                                  input_shape=(224, 224, 3)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Dodawanie kolejnych warstw konwolucyjnych
    for i in range(hp.Int('num_layers_conv', 1, 4)): #4
        hp_filters = hp.Int(f'conv_{i}_filters', min_value=64, max_value=192, step=32)  # 5
        model.add(keras.layers.Conv2D(filters=hp_filters, kernel_size=3, activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Flatten())

    # Dodawanie warstw gęstych oraz warstwy Dropout
    for i in range(hp.Int('num_layers_dense', 1, 3)):  # 3
        hp_units = hp.Int(f'units_{i}', min_value=64, max_value=192, step=64)  # 3
        model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))

        hp_dropout = hp.Float(f'dropout_rate_{i}', min_value=0.0, max_value=0.2, step=0.1)  # 3
        model.add(tf.keras.layers.Dropout(hp_dropout))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    # Optymalizacja stopnia nauki
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])  # 2
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Obliczenie wag klas na podstawie liczebności klas
class_weights = compute_class_weight('balanced', classes=np.unique(train_df['dx']), y=train_df['dx'])
class_weight_dict = dict(enumerate(class_weights))

# Zainicjowanie algorytmu Hyperband
hyperband = Hyperband(build_model,
                      objective='val_accuracy',
                      max_epochs=20,
                      factor=3,
                      directory='output2')

hyperband.search(train_generator,
                 epochs=20,
                 validation_data=validation_generator,
                 class_weight=class_weight_dict)

best_model = hyperband.get_best_models(num_models=1)[0]

print("\nResult summary : \n")
hyperband.results_summary()

best_model.save('Opt_Basic_Model.keras')

print("\nModel summary : \n")

best_model.summary()

# Pobieranie najlepszych hiperparametrów
best_hyperparameters = hyperband.get_best_hyperparameters(num_trials=1)[0]

# Wyświetlanie najlepszych hiperparametrów
print(best_hyperparameters.values)

# Przygotowanie danych testowych
test_images, test_labels = next(iter(test_generator))

# Wykonanie predykcji
predictions = best_model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

# Konwersja test_labels
if test_labels.ndim > 1:
    test_labels = np.argmax(test_labels, axis=1)

# Obliczenie F1 Score
f1 = f1_score(test_labels, predicted_classes, average='macro')
print(f"F1 Score dla jednego batcha: {f1}")

# Obliczenie i wyświetlenie raportu klasyfikacji
report = classification_report(test_labels, predicted_classes, target_names=unique_labels)
print(report)
