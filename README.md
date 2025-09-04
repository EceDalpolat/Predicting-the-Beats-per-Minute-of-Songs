# Predicting the Beats-per-Minute of Songs

Bu proje, şarkıların çeşitli müzik özelliklerinden yola çıkarak Beats-per-Minute (BPM) değerini tahmin etmeye yönelik bir makine öğrenmesi uygulamasıdır.

## Proje Yapısı

- `data/` : Ham ve işlenmiş veri dosyaları
- `src/` : Tüm Python kaynak kodları
    - `preprocessing/` : Veri temizleme ve özellik mühendisliği
    - `models/` : Model eğitimi ve değerlendirme
    - `submission/` : Tahmin ve sonuç dosyası oluşturma
    - `utils/` : Yardımcı fonksiyonlar ve ayarlar
- `docker/` : Docker ile ilgili dosyalar
- `venv/` : Sanal Python ortamı (gitignore ile hariç tutulur)

## Kullanım

1. **Veri Hazırlama:**
   - Ham veriyi `data/raw/` klasörüne ekleyin.
   - `src/preprocessing/clean_data.py` ve `src/preprocessing/feature_engineering.py` dosyalarını çalıştırarak veriyi temizleyin ve özellikleri oluşturun.

2. **Model Eğitimi:**
   - `src/models/train_regression.py` dosyası ile regresyon modellerini eğitin ve kaydedin.

3. **Tahmin ve Submission:**
   - `src/submission/make_submission.py` dosyası ile test verisi üzerinde tahmin yapıp submission dosyasını oluşturun.

## Gereksinimler

- Python 3.9+
- Gerekli paketler için: `requirements.txt`
- scikit-learn, pandas, numpy, joblib vb.

## Notlar
- Tüm veri ve model dosyaları `.gitignore` ile hariç tutulmuştur.
- Docker desteği mevcuttur.

## Katkı
Pull request ve issue açarak katkıda bulunabilirsiniz.

## Lisans
MIT
