import tensorflow as tf
import keras_cv
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import kagglehub
import os
import shutil
import matplotlib.pyplot as plt
import math

# ==========================================================================
# 1. CONFIGURACIÓN FINAL CON CLASES SIMPLIFICADAS
# ==========================================================================
class Config:
    IMG_SIZE = 224
    BATCH_SIZE = 32
    
    # +++ CAMBIO CLAVE: Definimos las nuevas clases y las fusiones +++
    CLASSES_TO_MERGE_GRANDE = ['E_pesado', 'bus']
    NEW_CLASS_GRANDE = 'Grande'
    
    CLASSES_TO_MERGE_CAMIONETA = ['jeep', 'SUV']
    NEW_CLASS_CAMIONETA = 'Camioneta'

    FINAL_CLASSES = ['Grande', 'Camioneta', 'family sedan'] # Nuestras 3 nuevas clases
    NUM_CLASSES = len(FINAL_CLASSES)
    
    LR_START = 1e-4
    LR_FINETUNE = 5e-6
    
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 5

CONFIG = Config()

# ==========================================================================
# 2. PREPARACIÓN DE DATOS (CON DOBLE FUSIÓN)
# ==========================================================================
def reorganize_data(base_dir, classes_to_merge, new_class_name):
    new_class_path = os.path.join(base_dir, new_class_name)
    os.makedirs(new_class_path, exist_ok=True)
    print(f"Creando la clase '{new_class_name}' y moviendo archivos...")
    for class_name in classes_to_merge:
        original_class_path = os.path.join(base_dir, class_name)
        if not os.path.exists(original_class_path):
            print(f"Advertencia: La carpeta '{class_name}' no existe y será omitida.")
            continue
        for filename in os.listdir(original_class_path):
            shutil.move(os.path.join(original_class_path, filename), new_class_path)
        try: os.rmdir(original_class_path)
        except OSError: pass
    print(f"Fusión para '{new_class_name}' completada.")

try:
    path = kagglehub.dataset_download("marquis03/vehicle-classification")
    train_dir = os.path.join(path, 'train')
    # +++ CAMBIO CLAVE: Realizamos las dos fusiones que propusiste +++
    reorganize_data(train_dir, CONFIG.CLASSES_TO_MERGE_GRANDE, CONFIG.NEW_CLASS_GRANDE)
    reorganize_data(train_dir, CONFIG.CLASSES_TO_MERGE_CAMIONETA, CONFIG.NEW_CLASS_CAMIONETA)
except Exception as e:
    print(f"Error: {e}"); exit()

# ==========================================================================
# 3. GENERADORES DE DATOS
# ==========================================================================
datagen = ImageDataGenerator(
    rescale=1./255, validation_split=0.2, rotation_range=25, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.25,
    horizontal_flip=True, fill_mode='nearest'
)
train_generator = datagen.flow_from_directory(
    train_dir, target_size=(CONFIG.IMG_SIZE, CONFIG.IMG_SIZE), batch_size=CONFIG.BATCH_SIZE,
    class_mode='categorical', classes=CONFIG.FINAL_CLASSES, subset='training', shuffle=True
)
validation_generator = datagen.flow_from_directory(
    train_dir, target_size=(CONFIG.IMG_SIZE, CONFIG.IMG_SIZE), batch_size=CONFIG.BATCH_SIZE,
    class_mode='categorical', classes=CONFIG.FINAL_CLASSES, subset='validation', shuffle=False
)
from sklearn.utils.class_weight import compute_class_weight
# Con clases más balanceadas, el cálculo automático es suficiente y más seguro.
class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
class_weight_dict = dict(enumerate(class_weights))
print("\nPesos de clase para el nuevo problema:", class_weight_dict)

# ==========================================================================
# 4. CONSTRUCCIÓN DEL MODELO
# ==========================================================================
base_model = MobileNetV2(
    input_shape=(CONFIG.IMG_SIZE, CONFIG.IMG_SIZE, 3), include_top=False, weights='imagenet'
)
base_model.trainable = False
inputs = Input(shape=(CONFIG.IMG_SIZE, CONFIG.IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.6)(x)
# La capa de salida ahora se ajusta automáticamente a 3 clases
predictions = Dense(CONFIG.NUM_CLASSES, activation='softmax')(x)
model = Model(inputs, predictions)

# Mantenemos FocalLoss con gamma alto, que funcionó muy bien
loss_function = keras_cv.losses.FocalLoss(from_logits=False, gamma=3.0)

# ==========================================================================
# 5. ENTRENAMIENTO Y AJUSTE FINO (La estrategia no cambia)
# ==========================================================================
print("\n--- Fase 1: Entrenamiento de la cabeza ---")
optimizer_head = keras.optimizers.AdamW(learning_rate=CONFIG.LR_START, weight_decay=1e-5)
model.compile(optimizer=optimizer_head, loss=loss_function, metrics=['accuracy'])
callbacks_list = [
    EarlyStopping(monitor='val_loss', mode='min', patience=CONFIG.EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=CONFIG.REDUCE_LR_PATIENCE, min_lr=1e-7, verbose=1)
]
history = model.fit(
    train_generator, epochs=150, validation_data=validation_generator,
    callbacks=callbacks_list, class_weight=class_weight_dict
)

print("\n--- Fase 2: Ajuste Fino ---")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False
optimizer_finetune = keras.optimizers.AdamW(learning_rate=CONFIG.LR_FINETUNE, weight_decay=1e-6)
model.compile(optimizer=optimizer_finetune, loss=loss_function, metrics=['accuracy'])
if len(history.epoch) < 150:
    history_fine = model.fit(
        train_generator, epochs=len(history.epoch) + 150, initial_epoch=len(history.epoch),
        validation_data=validation_generator, callbacks=callbacks_list, class_weight=class_weight_dict
    )
else:
    history_fine = None

# ==========================================================================
# 6. VISUALIZACIÓN Y DIAGNÓSTICO FINAL CON TTA
# ==========================================================================
# ... (Esta sección no cambia) ...
print("\nGenerando gráficas del historial de entrenamiento...")
acc = history.history.get('accuracy', [])
val_acc = history.history.get('val_accuracy', [])
loss = history.history.get('loss', [])
val_loss = history.history.get('val_loss', [])
if history_fine:
    acc += history_fine.history.get('accuracy', [])
    val_acc += history_fine.history.get('val_accuracy', [])
    loss += history_fine.history.get('loss', [])
    val_loss += history_fine.history.get('val_loss', [])
epochs_range = range(len(acc))

if epochs_range:
    plt.figure(figsize=(14, 7)); plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Precisión Entrenamiento'); plt.plot(epochs_range, val_acc, label='Precisión Validación')
    if history_fine: plt.axvline(x=len(history.epoch)-1, color='grey', linestyle='--', label='Inicio Ajuste Fino')
    plt.legend(loc='lower right'); plt.title('Precisión'); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Pérdida Entrenamiento'); plt.plot(epochs_range, val_loss, label='Pérdida Validación')
    if history_fine: plt.axvline(x=len(history.epoch)-1, color='grey', linestyle='--', label='Inicio Ajuste Fino')
    plt.legend(loc='upper right'); plt.title('Pérdida'); plt.grid(True)
    plt.savefig('training_history_simplified.png'); plt.show()

print("\n--- Iniciando Diagnóstico del Modelo Final con Test Time Augmentation (TTA) ---")
tta_steps = 5
predictions_tta = []
for i in range(tta_steps):
    print(f"Paso TTA {i+1}/{tta_steps}...")
    validation_generator.reset()
    steps = math.ceil(validation_generator.samples / validation_generator.batch_size)
    preds = model.predict(validation_generator, steps=steps)
    predictions_tta.append(preds)

avg_predictions = np.mean(predictions_tta, axis=0)
predicted_classes = np.argmax(avg_predictions, axis=1)
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

print('\n--- Reporte de Clasificación Detallado (con TTA) ---')
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Matriz de Confusión (con TTA)'); plt.ylabel('Clase Verdadera'); plt.xlabel('Clase Predicha'); plt.show()


# ==========================================================================
# 7. GUARDADO FINAL DEL MODELO
# ==========================================================================
model.save('modelo_simplificado_final.h5')
print('\nModelo guardado como modelo_simplificado_final.h5')