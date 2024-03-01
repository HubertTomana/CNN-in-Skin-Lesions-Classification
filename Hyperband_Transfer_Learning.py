from keras import layers
import os
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense
from keras.optimizers import Adam
import pandas as pd
from kerastuner.tuners import Hyperband
from sklearn.model_selection import train_test_split
from keras.layers import Dropout

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

# Wczytanie modelu do uczenia transferowego
base_model = MobileNet(include_top=False, input_shape=(224, 224, 3))

# Zamrozenie wag
base_model.trainable = False

# Model do algorytmu Hyperband
def build_model(hp):
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
    ])
    model.add(Dense(units=160, activation='relu')),
    model.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1))),

    model.add(Dense(num_classes, activation='softmax'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Obliczenie wag klas na podstawie liczebności klas
class_weights = compute_class_weight('balanced', classes=np.unique(train_df['dx']), y=train_df['dx'])
class_weight_dict = dict(enumerate(class_weights))

# Zainicjowanie algorytmu Hyperband
tuner = Hyperband(build_model,
                  objective='val_accuracy',
                  max_epochs=20,
                  factor=4,
                  directory='output',
                  project_name='SkinCancerClassificationHyperband')

tuner.search(train_generator,
             epochs=20,
             validation_data=validation_generator,
             class_weight=class_weight_dict)

best_model = tuner.get_best_models(num_models=1)[0]

print("\nResult summary : \n")
tuner.results_summary()

best_model.save('Opt_Mobilenet.keras')

print("\nModel summary : \n")

best_model.summary()

# Pobieranie najlepszych hiperparametrów
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

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
