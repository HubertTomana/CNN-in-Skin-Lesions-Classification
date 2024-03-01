from keras.models import Sequential
from keras import layers
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

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

# Implementacja splotowej sieci neuronowej
model = Sequential([
    layers.Conv2D(64, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(192, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(160, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(192, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(num_classes, activation='softmax')  # Używamy softmax dla wielu klas
])

# Proba implementacji wlasnej metryki

# class Metrics(Callback):
#     def __init__(self, validation_generator):
#         super(Metrics, self).__init__()
#         self.validation_generator = validation_generator
#
#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         val_targ = []
#         val_predict = []
#
#         for batch in range(len(self.validation_generator)):
#             xVal, yVal = next(self.validation_generator)
#             val_targ.extend(yVal)
#             val_predict.extend(np.argmax(self.model.predict(xVal), axis=-1))
#
#         val_targ = np.array(val_targ)
#         val_predict = np.array(val_predict)
#
#         _val_f1 = f1_score(val_targ, val_predict, average='macro')
#         _val_recall = recall_score(val_targ, val_predict, average='macro')
#         _val_precision = precision_score(val_targ, val_predict, average='macro')
#
#         logs['val_f1'] = _val_f1
#         logs['val_recall'] = _val_recall
#         logs['val_precision'] = _val_precision
#
#         print(f" — val_f1: {_val_f1:.4f} — val_precision: {_val_precision:.4f} — val_recall: {_val_recall:.4f}")
#         return
# metrics = Metrics(validation_generator)


# Obliczenie wag klas na podstawie liczebności klas
class_weights = compute_class_weight('balanced', classes=np.unique(train_df['dx']), y=train_df['dx'])
class_weight_dict = dict(enumerate(class_weights))

# Kompilacja modelu
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

history = model.fit(train_generator,
                    epochs=15,
                    validation_data=validation_generator,
                    class_weight=class_weight_dict)

model.save('Final_Basic_Model.keras')

print("\nModel summary : \n")

model.summary()

# Przygotowanie danych testowych
test_images, test_labels = next(iter(test_generator))

# Wykonanie predykcji
predictions = model.predict(test_images)
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
