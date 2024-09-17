# Prinsip Deep Q-Learning

Telah kita ketahui bahwa salah satu kelemahan Q-learning adalah memori dan waktu karena harus mempelajari dan menyimpan Q-value dari setiap state dan action. Bila kita batasi proses pembelajaran agent tersebut untuk menghemat waktu dan memori, maka tidak semua kemungkinan dicoba. Saat agent tersebut selesai belajar dan dijalankan terdapat kemungkinan agent tersebut menemukan state baru. Akibatnya agent tersebut tidak tahu harus berbuat apa. Dengan kata lain, kekurangan Q-learning adalah tidak melakukan generalisasi terhadap state dan action yang mungkin.

Namun bagaimana bila kita bisa mengestimasi berbagia kemungkinan Q-value? Itulah yang dilakukan oleh Deep Q-learning. Deep Q-learning berusaha mengestimasi Q-value dari action yang diambil untuk setiap state yang ada. Input dari Deep Q-learning adalah gambar state saat ini.\*\*\*\*
