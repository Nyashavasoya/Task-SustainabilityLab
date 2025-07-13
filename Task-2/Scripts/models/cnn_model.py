from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, BatchNormalization, GlobalMaxPooling1D

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        Conv1D(32, kernel_size=2, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        GlobalMaxPooling1D(),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model