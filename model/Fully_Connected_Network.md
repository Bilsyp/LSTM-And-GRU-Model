# Fully Connected Network (FCN)

Fully Connected Network (FCN) adalah jenis arsitektur neural network yang setiap neuron pada layer sebelumnya terhubung dengan setiap neuron pada layer berikutnya. Dalam FCN, informasi mengalir dari input layer ke hidden layers dan akhirnya ke output layer. Setiap neuron pada layer terhubung dengan semua neuron pada layer berikutnya, sehingga informasi dapat tersebar dengan baik di seluruh jaringan.

Perbedaan antara FCN dengan Convolutional Neural Network (CNN) dan Recurrent Neural Network (RNN) adalah sebagai berikut:

- FCN: Setiap neuron pada layer terhubung dengan semua neuron pada layer berikutnya. Cocok untuk data yang tidak memiliki struktur spasial atau urutan tertentu.
- CNN: Memiliki layer konvolusi yang memungkinkan jaringan untuk mengekstrak fitur spasial dari data, seperti gambar. Cocok untuk data dengan struktur spasial.
- RNN: Dirancang untuk menangani data berurutan, seperti teks atau time series. Memiliki kemampuan untuk "mengingat" informasi dari waktu sebelumnya.

FCN termasuk dalam kategori Deep Learning, yang merupakan subset dari Machine Learning. Deep Learning menggunakan neural network dengan banyak layer (dalam hal ini, FCN) untuk mempelajari representasi data yang kompleks.

![fnc](../images/fcn.png)
![rnn](../images/rnn.png)
![cnn](../images/cnn.png)

## Membuat Model FCN Untuk Memprediksi Bitrate dengan Tensorflow atau Keras

## Dataset Yang Digunakan , Dengan Panjang atau Jumlah 200 lebih.
![dataset](../images/dataset.png)

## Hasil Menunjukkan Model Mampu Memprediksi 80%

![dataset](../images/output.png)

## Next Step

```json
Machine Learning
└── Supervised Learning
    ├── Classification
    └── Regression
└── Unsupervised Learning
    ├── Clustering
    ├── Dimensionality Reduction
    ├── Association Rule Learning
    └── Anomaly Detection

Deep Learning (Subset dari Machine Learning)
└── Supervised Learning
    ├── CNN (untuk gambar)
    ├── RNN (untuk urutan data)
    └── Fully Connected Networks (untuk berbagai aplikasi)
└── Unsupervised Learning
    ├── Autoencoders
    └── GANs

```

