import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Cargar el modelo
model = load_model('red3.h5')  # Nombre del modelo


# Crear un objeto para acceder a la cámara 
cap = cv2.VideoCapture(0)

while True:
    # Captura un fotograma de la cámara
    ret, frame = cap.read()
    
    # Realiza preprocesamiento de la imagen si es necesario
    frame = cv2.resize(frame, (224, 224))  # Tamaño de entrada del modelo
    frame = frame / 255.0  # Normalizar los valores de píxeles
    
    # Realiza inferencia con el modelo
    predictions = model.predict(np.expand_dims(frame, axis=0))
    
    # Obtener la clase con la probabilidad más alta
    predicted_class = np.argmax(predictions)
    
    # Mostrar el resultado en el fotograma de la cámara
    cv2.putText(frame, f'Clase: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Mostrar el fotograma de la cámara en una ventana
    cv2.imshow('Clasificador de Señales de Tráfico', frame)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()