# PROJE:
#########
# Ürün ratinglerini daha doğru hesaplamaya çalışmak ve ürün yorumlarını daha doğru sıralamak.


#### Veri Seti Hikayesi #####
# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.


##### Değişkenler
# reviewerID – Kullanıcı ID’si Örn: A2SUAM1J3GNN3B
# asin – Ürün ID’si. Örn: 0000013714
# reviewerName – Kullanıcı Adı
# helpful – Faydalı değerlendirme derecesi
# reviewText – Değerlendirme Kullanıcının yazdığı inceleme metni
# overall – Ürün rating’i
# summary – Değerlendirme özeti
# unixReviewTime – Değerlendirme zamanı Unix time
# reviewTime – Değerlendirme zamanı Raw
# day_diff – Değerlendirmeden itibaren geçen gün sayısı helpful_yes – Değerlendirmenin faydalı bulunma sayısı
# total_vote – Değerlendirmeye verilen oy sayısı


import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

from google.colab import files
df_= files.upload()
#veri dosyası yükleme

df = pd.read_csv("amazon_review.csv")  
#dosya okuma


df.head()

df.tail()

df["overall"].value_counts()
# puanı veren kişi sayısı

df["day_diff"].describe().T
# en son degerendirme 1 gün önce
# ilk degerlendirme 1064 gün önce

df["overall"].mean()
# var olan average rating

"""GÖREV1
*  Average Rating’i güncel yorumlara göre
hesaplayınız.
 ( Güncel yorumlara göre sıralamak için zaman agırlıklı average rating alırım.)
 * * day_diff: degerlendirmenin üzerinden gecen gün sayısı
 * * overall: ürün ratingi
"""

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= 30, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 90), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 180), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 180), "overall"].mean() * w4 / 100

# güncel yorumlara agırlık verilmesini istedigim için 30 günlük yorumlara daha fazla oran tanımlıyorum.

time_based_weighted_average(df)

# güncel yorumlara göre ürün ratingi aldıgımızda görüyoruz ki, ürün puanı daha yüksek.
# ürünün güncel ratingi var olan ratinge göre daha anlamlıdır.

"""GÖREV2
* Sorting Reviews
"""

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["helpful_no"]= df['total_vote']-df['helpful_yes']

df["wilson_lower_bound_score"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]),  axis=1)

df.sort_values("wilson_lower_bound_score", ascending=False).head(20)

