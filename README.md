
# SAAT MODEL TESPİTİ

  ## Giriş

Bu notebookta 10 tane kol saatinin resimlerinden küçük bir veri seti oluşturdum. Veri setinde train,validation, test  isimli 3 klasör bulunuyor. Her bir saatten 12 adet resim bulunmaktadır. Resimler internetten elde edilmiştir. 8 resim eğitim ,2 resim doğrulama ve 2 resimde test için kullanılmıştır. Amacımız saat resimlerinden saatin hangi marka ve model olduğunu tespit etmektir.
### Emporio Armani - Ar1971 <img src="saat.jpg" width="100">

## Kütüphaneler
Uygulamımızı geliştirirken model oluşturmayı ve eğitmeyi oldukça basit hale getiren bir deep learning kütüphanesi olan **Keras**  kullandım. Keras ile ilgili dökümanlara [Buradan](https://keras.io/)  ulaşabilirsiniz.

```ruby
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
```


