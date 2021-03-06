
# SAAT MODEL TESPİTİ

  ## Giriş

Bu notebookta 10 tane kol saatinin resimlerinden küçük bir veri seti oluşturdum. Veri setinde train,validation, test  isimli 3 klasör bulunuyor. Her bir saatten 12 adet resim bulunmaktadır. Resimler internetten elde edilmiştir. 8 resim eğitim ,2 resim doğrulama ve 2 resimde test için kullanılmıştır. Amacımız saat resimlerinden saatin hangi marka ve model olduğunu tespit etmektir.
### Emporio Armani - Ar1971 <img src="saat.jpg" width="100">

## Kütüphaneler
Uygulamımızı geliştirirken model oluşturmayı ve eğitmeyi oldukça basit hale getiren bir deep learning kütüphanesi olan **Keras**  kullandım. Keras ile ilgili dökümanlara [Buradan](https://keras.io/)  ulaşabilirsiniz.

```python
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

```python
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

```python
#deneme
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
İşin en önemli kısmı burasıdır. Tecrübe ve sezgiden faydalanarak verimize en uygun model seçilir. Çeşitli modeller üzerinde deneyler yapılır ve en uygun model tasarlanır. Modelimiz Convolutional olarak tasarlanmıştır . CNN resim işlemlerinde oldukça başarılı ve hızlıdır. Modelimizde işlem karmaşıklığını azaltmak(boyutları küçültmek) için max pooling katmanları kullanılmıştır. CNN' de overfitting problemlerini önlemek için dropout layerlarından faydalanılmıştır. Modelimizdeki parametreler önemlidir. Ara katmanlarda relu aktivasyon fonksiyonu kullanırken çıkış katmanında softmax fonksiyonu kullanılmıştır. Verilerimiz kategorik olarak sınıflandırılmak istendiğinden loss fonksiyonu "categorical_crossentropy" seçilmiştir. optimazyon algoritması olarak ta momentum ve rsmpprop 'u bir arada uygulayan "adam" seçilmiştir. Modelinizin katman sayısı, ara katmanların boyutları,loss fonksiyonu, aktivasyon fonksiyonları modelinizin başarısını etkileyecektir. Bunların titizlikle tespit edilmesi gerekir.
```python
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

#Modelimizi fit ederek eğitime başlayabiliriz.
```python
H=model.fit_generator( 
    training_batch, 
    steps_per_epoch=300, #her çağda  kaç örnek türeteceğimizi belirliyoruz.
    epochs=epochs, # kaç çağ olacağını belirtiyoruz yukarıda 10 olarak belirtmiştik.
    validation_data=validation_batch, 
    validation_steps=10
    ) 
 ```
 <img src="accuracy_per_epoch.png">
## Test ve Hata Matrisi(confusion matrix)
<img src="confusion.png">
Çıktının birden fazla olduğu durumlarda modelin performansını görmemize yarar. Köşegendeki değerler doğru tahminleri gösterir.Köşegen dışında kalan değerler yanlışları gösterir. Eğittiğimiz model ile test verilerini predict edip hata matrisini çizdiriyoruz.

```python
x_test,y_test=test_batch.next()
y_pred_test=model.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test.argmax(axis=1), y_pred_test.argmax(axis=1))
print(cm)
print(H.history.keys())
 ```
 ### Sonuç
%90 üzeri başarı ile 10 saati tespit edebiliyoruz. Kaynakların kısıtlı olmasından dolayı saat sayısı ve dataset küçük seçildi. İmkanlar ve veri bulnuabilmesi durumunda daha büyük veriler üzerinde uygulanıp geliştirilebilir.Test verilerimizi predict ettikten sonra resimlerle beraber sonuçları göstermek için fonksiyonumuzu tanımlıyoruz. Verileri görselleştirmek için matplotlib kütüphanesinden faydalanıyoruz. 
```python
def show_images(images, cols = 1, titles = None,labels=None):
    """Resimleri figürde gösterir..
    
    Parameters
    ---------
    images: np.arrays dizisidir.
    
    cols (Default = 1): figürdeki kolon sayısıdır.(n_images/float(cols))).
    
    titles: Her resmin başlığıdır.
    """
    assert((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title('tahmin:' + str(titles[n]))
        a.set_ylabel('model:' + str(labels[n]))
        # Hide grid lines
     
        
        # Hide axes ticks
        a.set_xticks([])
        a.set_yticks([])
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images/2)
    plt.show()

   
show_images(x_test, cols = 1, titles =y_pred_test.argmax(axis=1) ,labels=y_test.argmax(axis=1))
```
<img src="examples.png">   
 
## Modelin ve Ağırlıkların Kaydedilmesi.
```python
  model.save_weights('model1_weights.h5')
     # serialize model to JSON
     model_json = model.to_json()
     with open("model1.json", "w") as json_file:
     json_file.write(model_json)
 ```    
   
