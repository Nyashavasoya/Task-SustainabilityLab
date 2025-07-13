from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, GlobalMaxPooling1D, ConvLSTM2D, Reshape

def build_convlstm_model(input_shape, num_classes):
    # have to reshape input for ConvLSTM: (samples, timesteps, rows, cols, channels)
    model = Sequential([
        Reshape((1, 4, 1, 1), input_shape=input_shape),
        ConvLSTM2D(32, (2, 1), activation='relu', return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        ConvLSTM2D(16, (2, 1), activation='relu', return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        Flatten(),
        Dense(64, activation='relu'),
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