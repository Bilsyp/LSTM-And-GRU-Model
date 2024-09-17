```bash
model.add(GRU(units=128, return_sequences=True,
              kernel_regularizer=l2(0.001),   # Tambahkan L2 regularization
              input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(GRU(units=64, kernel_regularizer=l2(0.002)))  # L2 lagi di layer GRU kedua
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
```

-0.2

```bash
model.add(GRU(units=64, return_sequences=True,
              kernel_regularizer=l2(0.001),   # Tambahkan L2 regularization
              input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(GRU(units=64, kernel_regularizer=l2(0.004)))  # L2 lagi di layer GRU kedua
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
history = model.fit(X_train, y_train, epochs=100,batch_size=4, callbacks=[early_stopping],validation_split=0.2, verbose=1)
```

0.004

```bash
model.add(GRU(units=64, return_sequences=True,
              kernel_regularizer=l2(0.001),   # Tambahkan L2 regularization
              input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.1))
model.add(GRU(units=128, kernel_regularizer=l2(0.003)))  # L2 lagi di layer GRU kedua
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()****

0.1

```
    N      : Jumlah total data (rows/samples)
    T      : Panjang sequence (sequence length)
    n      : Ukuran batch (batch size)
    h1     : Hidden units untuk LSTM layer pertama
    h2     : Hidden units untuk LSTM layer kedua
    epochs : Jumlah epochs untuk training