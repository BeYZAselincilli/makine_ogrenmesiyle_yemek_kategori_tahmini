import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Excel dosyasının yolu
excel_path = r"excel dosyasının yolu"

# Veriyi yükle
veri = pd.read_excel(excel_path)

def compute_safety_score(item_name, madde_grubu_adi):
    # Özellikleri vektörleştir
    X = veri['ITEMNAME'] + " " + veri['MADDE_GRUBU_ADI']
    y = veri['LEVEL_2']

    vektorlestirme = CountVectorizer()
    X_vektorlestirilms = vektorlestirme.fit_transform(X)

    # Sınıflandırıcıyı eğit
    siniflandirma = MultinomialNB()
    siniflandirma.fit(X_vektorlestirilms, y)

    # Veriyi vektörleştir ve tahmin et
    veri_vektor = vektorlestirme.transform([item_name + " " + madde_grubu_adi])
    tahmin_prob = siniflandirma.predict_proba(veri_vektor)

    # Tahmin edilen sınıfı ve güvenlik oranını hesapla
    tahmin_sinifi = siniflandirma.classes_[tahmin_prob.argmax()]

    diger_siniflar = siniflandirma.classes_[siniflandirma.classes_ != tahmin_sinifi]
    diger_olasiliklar = tahmin_prob[0, siniflandirma.classes_ != tahmin_sinifi]

    guvenlik_orani = (tahmin_prob.max() - diger_olasiliklar.max()) / diger_olasiliklar.max() * 100

    return tahmin_sinifi, guvenlik_orani

if __name__ == "__main__":
    item_name = input("Ürün Adı (ITEMNAME): ")
    madde_grubu_adi = input("Madde Grubu Adı (MADDE_GRUBU_ADI): ")

    tahmin_sinifi, guvenlik_orani = compute_safety_score(item_name, madde_grubu_adi)

    print("Tahmin Edilen LEVEL_2:", tahmin_sinifi)
    print("KATEGORİNİN GÜVENLİK ORANI %", guvenlik_orani)

    # Alternatif tahminleri hesapla
    X = veri['ITEMNAME'] + " " + veri['MADDE_GRUBU_ADI']
    y = veri['LEVEL_2']
    vektorlestirme = CountVectorizer()
    X_vektorlestirilms = vektorlestirme.fit_transform(X)
    siniflandirma = MultinomialNB()
    siniflandirma.fit(X_vektorlestirilms, y)
    veri_vektor = vektorlestirme.transform([item_name + " " + madde_grubu_adi])
    tahmin_prob = siniflandirma.predict_proba(veri_vektor)
    diger_siniflar = siniflandirma.classes_[siniflandirma.classes_ != tahmin_sinifi]
    diger_olasiliklar = tahmin_prob[0, siniflandirma.classes_ != tahmin_sinifi]
    en_yakin_alternatifler = sorted(zip(diger_siniflar, diger_olasiliklar), key=lambda x: x[1], reverse=True)[:5]

    print("\nEn Yakın 5 Alternatif Tahmin:")
    for sinif, olasilik in en_yakin_alternatifler:
        print(f"{sinif}: %{olasilik*100:.2f}")

    # Tüm alternatiflerin olasılığının toplamını hesapla
    toplam_olasilik = sum(olasilik for _, olasilik in en_yakin_alternatifler)

    print("toplam alt kategorilerin oranı toplamı %", toplam_olasilik*100 )

    print(f"\nBu tahmin, diğer sınıflara göre %{guvenlik_orani:.2f} daha güvenilir.")
