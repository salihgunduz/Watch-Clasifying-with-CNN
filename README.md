
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

## CPU yerine GPU kullanımı
Uygulamamız görüntüler üzerinde çalışacağı için oldukça fazla işlem gücüne iytiyaç duyacağız. Ağımıza girdi olarak 100 px * 100px görüntüler vereceğiz. Görüntünün RGB kanallarını, örnek sayımızı da düşünürsek girdimiz 100x100x3xörnek_sayısı boyutunda olacaktır. Bir de epoch(çağ) dediğimiz döngü sayısını da hesaba katarsak bu kadar küçük bir veri setinde bile oldukça fazla işlem yapılacaktır. Bu işlemleri daha hızlı yapabilmek için CPU kullanmak yerine bilgisayarımızın paralel işlem gücü daha yüksek olan GPU'sunu kullanacağız. Bunu yapabilmek için Tensorflow GPU kurulumu yapmamız gerekiyor. Siz de ekran kartınızın Cuda Toolkiti destekleyelip destelemediğine [Buradan](https://developer.nvidia.com/cuda-gpus) bakabilirsiniz. Aşağıda gerekli kütüphaneyi çağırıp Nvidia ekran kartımı eğitim için belirledim.

# GPU seçimi yapıyoruz.
```ruby
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))
```

## Parametrelerin Belirlenmesi
İşlem gücünü hafifletmek ve öğrenmeyi hızlandırmak için verilerimizi batch dediğimiz demetlere bölebiliriz. Görüntü boyutu 100x100 olarak ayarladım. Ağınızın perfomansına ve doğruluk oranına göre değiştirebilirsiniz. Doğru sonuca ulaşabiliyorsanız işlem hızını arttırmak açısından mümkün olduğunda küçük tutabiliriz. Epoch sayısı da yine denenerek bulunabilir. Epoch değerini yüksek tutmak eğitim süresini uzattığı gibi ağın ezberlemesine de (overfitting) sebep olabilir. Deneyerek uygun değeri bulabiliriz.

```ruby
train_batch_size=64 # Ekran kartımızın kaldırabileceği boyutta seçilmelidir. Aksi takdirde hata alırız.
validation_batch_size=64
image_width=100 # resim genişliği
image_height=100 #resim yüksekliği
epochs=10  #çağ sayısı
number_of_classes=10  #sınıf sayısı tespit edeceğimiz saat sayısı 10'dur.
```

## Keras Data Augmentation (Veri Çoğaltma)
Verilerin azlığı overfitting (ezberlerme) probeleminin en önemli nedenlerinden birisidir. Eğitimde kullanılan 8'er adet resmin  için yeterli olmadığı görülmüştür. Bu yüzden verileri arttırmaya karar verdim. internette yeteri kadar görüntü bulamadım. bu yüzden keras'ın veri arttırma yöntemlerini kullandım. Keras veri arttırma ile ilgili bilgilere [Buradan](https://keras.io/preprocessing/image/)  ulaşabilirsiniz.
<img src="enlarge1.jpg" width="500">

```ruby
#resimleri yüklüyoruz.
#resimleri klasörlerinden okuyarak data generator sayesinde çoğaltıp kullanacağımız değişkenlere atıyoruz.
#resimleri döndürerek, kaydırarak, ışık ayarları ile oynayarak yeni resimler türetiyoruz.
# Resimleri Çok fazla bozmamaya dikkat edelim.
train_datagen = ImageDataGenerator( 
        rescale=1. / 255,  #Normalizasyon yapıyoruz.
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=(0.5,1.0),
        zoom_range=0.1)
test_datagen = ImageDataGenerator(rescale=1. / 255)


training_batch = train_datagen.flow_from_directory('new_data/train',
                                             target_size = (image_width, image_height),
                                             batch_size = train_batch_size,
                                             class_mode = 'categorical',
                                             shuffle=True
                                             )
validation_batch=test_datagen.flow_from_directory('new_data/validation',
                                             target_size = (image_width, image_height),
                                             batch_size = validation_batch_size,
                                             class_mode = 'categorical',
                                             shuffle=True
                                            )
test_batch=test_datagen.flow_from_directory('new_data/test',
                                             target_size = (image_width, image_height),
                                             batch_size = validation_batch_size,
                                             class_mode = 'categorical',
                                             shuffle=True
                                            )
```

## Modelimizin Oluşturulması
İşin en önemli kısmı burasıdır. Tecrübe ve sezgiden faydalanarak verimize en uygun model seçilir. Çeşitli modeller üzerinde deneyler yapılır ve en uygun model tasarlanır. Modelimiz Convolutional olarak tasarlanmıştır . CNN resim işlemlerinde oldukça başarılı ve hızlıdır. Modelimizde işlem karmaşıklığını azaltmak(boyutları küçültmek) için max pooling katmanları kullanılmıştır. CNN' de overfitting problemlerini önlemek için dropout layerlarından faydalanılmıştır. Modelimizdeki parametreler önemlidir. Ara katmanlarda relu aktüvasyon fonksiyonu kullanırken çıkış katmanında softmax fonksiyonu kullanılmıştır. Verilerimiz kategorik olarak sınıflandırılmak istendiğinden loss fonksiyonu "categorical_crossentropy" seçilmiştir. optimazyon algoritması olarak ta momentum ve rsmpprop 'u bir arada uygulayan "adam" seçilmiştir. Modelinizin katman sayısı, ara katmanların boyutları,loss fonksiyonu, aktivasyon fonksiyonları modelinizin başarısını etkileyecektir. Bunların titizlikle tespit edilmesi gerekir.
```ruby
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(image_width,image_height,3)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=(image_width,image_height,3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(number_of_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
```
```ruby
#Modelimizi fit ederek eğitime başlayabiliriz.
H=model.fit_generator( 
    training_batch, 
    steps_per_epoch=300, #her çağda  kaç örnek türeteceğimizi belirliyoruz.
    epochs=epochs, # kaç çağ olacağını belirtiyoruz yukarıda 10 olarak belirtmiştik.
    validation_data=validation_batch, 
    validation_steps=10
    ) 
 ```
