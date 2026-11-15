# Deney sonuçlarına ilişkin notlar

**Görüntü İşleme ara sınavı — 264309500**

Bu notlarda, 15 Kasım 2026 tarihinde Fluo-N2DH-GOWT1 eğitim görüntülerinden elde ettiğim sayım sonuçlarını özetledim. Sayılar proje klasöründeki CSV dosyalarından alınmıştır; challenge'ın ayrı test arşivi kullanılmadı.

## Veri seçimi

Öğrenci numaramın son beş hanesi 09500 olduğu için seed değerini 9500 + 2026 = **11526** olarak belirledim. 01 sekansından 10 geliştirme ve 5 doğrulama karesi, 02 sekansından ise üçer ayrı blokta toplam 15 test karesi seçildi. Karelerin tamamı `selected_frames.csv` içinde bulunuyor.

## Geliştirme deneyleri

Toplam 24 konfigürasyonu 10 geliştirme görüntüsünde denedim. Başlangıçtaki ham Otsu yönteminin geliştirme MAE değeri 3,90'dı. Yüzdelik kontrast germe tek başına bu değeri 4,60'a çıkardı. Gaussian 5×5, aynı koşullardaki 3×3 filtreye göre küçük bir iyileşme sağladı (4,40'a karşı 4,60).

En belirgin farklardan biri opening işleminde ortaya çıktı: 3×3 opening için MAE 3,40, 5×5 opening için 2,70 hesaplandı. Histogram eşitleme ve adaptif eşikleme ise bu görüntülerde çok sayıda gereksiz bağlı bölge oluşturduğu için yüksek hata verdi. Deneylere ait tam tablo `experiments.csv` dosyasında.

Geliştirmedeki en iyi üç adayın doğrulama karşılaştırması:

| Yöntem | Doğrulama MAE | Doğrulama RMSE |
|---|---:|---:|
| E15 — opening 3×3 | 4,20 | 4,63 |
| E16 — opening 5×5 | 4,80 | 4,94 |
| E10 — sabit eşik 120 | 5,60 | 5,83 |

Geliştirmede E16 birinci olmasına rağmen doğrulamada E15 daha iyi çıktı. Bu nedenle **E15** seçildi: yüzde 2–98 kontrast germe → Gaussian 3×3 → Otsu → 3×3 opening → minimum 40 piksel alanlı bağlı bileşen sayımı.

## Bağımsız test

Test tahminlerini görmeden önce `final_pipeline.json` ve `test_predictions_locked.csv` dosyalarını kaydettim. Bu dosyalar ilk Git commit'inde saklandı. Kilit özeti `lock_manifest.json` içinde bulunuyor. Test referansları ancak kayıtlı içerik kontrol edildikten sonra okundu.

| Ölçüt | Başlangıç yöntemi | E15 |
|---|---:|---:|
| MAE | 3,4667 | 13,6667 |
| RMSE | 3,8471 | 13,8106 |
| Bias (tahmin − gerçek) | −1,8667 | +13,6667 |
| Tam doğru sayım oranı | %6,67 | %0,00 |
| Ortalama işlem süresi (s/kare) | 0,00762 | 0,01799 |

15 test karesinde referans hücre sayısı 23–30 arasında değişti. E15'in tahminleri ise 35–45 arasındaydı ve bütün karelerde fazla sayım yaptı. Blok MAE değerleri 13,60, 14,00 ve 13,40 olarak hesaplandı. Özellikle 022. karede 24 hücre varken E15 40 hücre saydı; aynı karede başlangıç yöntemi 24 buldu.

Bu sonuç, E15'in Sequence 01'deki başarısının Sequence 02'ye taşınmadığını gösteriyor. Bunun kontrast germe sonrasında küçük parçaların ayrı nesne sayılmasından kaynaklanması mümkün. Ancak burada piksel düzeyinde eşleştirme yapmadığım için bu açıklamayı kesin bir neden olarak değerlendirmiyorum. Testten sonra parametreleri değiştirmedim.

## Dosya karşılıkları

- Geliştirme: `experiments.csv` ve `development_per_frame.csv`
- Doğrulama: `validation_results.csv` ve `validation_per_frame.csv`
- Sabitlenen yöntem: `final_pipeline.json`
- Testten önce kaydedilen tahminler: `test_predictions_locked.csv`
- Test: `test_results.csv` ve `test_metrics.csv`
- Ortam ve seçim: `analysis_environment.json`, `selected_frames.csv`

Teknik PDF ve mikroskobik görüntüler özel ders teslimi için ayrıdır. GitHub klasöründe veri setinin TIFF dosyaları veya anotasyonları yer almaz. İki aşamalı Git geçmişi henüz GitHub'a aktarılmamıştır.
