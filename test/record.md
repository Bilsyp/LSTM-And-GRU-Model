```bash
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.add(GRU(units=128, return_sequences=True,
              kernel_regularizer=l2(0.001),    # Tambahkan L2 regularization
              input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dropout(0.1))
model.add(GRU(units=64 ,activation="relu", return_sequences=True, kernel_regularizer=l2(0.002)))
model.add(GRU(units=64 ,activation="relu",return_sequences=False))  # L2 lagi di layer GRU kedua
# L2 lagi di layer GRU kedua
model.add(Dense(1,activation="linear"))
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=["accuracy"])
model.summary()
```

## Percobaan ke dua

```bash
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.add(GRU(units=256, return_sequences=True, kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))  # Tambahkan dropout buat ngurangin overfitting
model.add(GRU(units=128, activation="relu", return_sequences=True, kernel_regularizer=l2(0.0015)))
model.add(Dropout(0.3))  # Dropout lagi di sini
model.add(GRU(units=64, activation="relu", return_sequences=False))
model.add(Dense(1, activation="linear"))
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=["accuracy"])

Hasil :accuracy: 0.0482 - loss: 0.0165 - val_accuracy: 0.0648
```
