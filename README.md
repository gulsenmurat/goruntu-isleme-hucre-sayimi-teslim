# Mikroskobik görüntülerde hücre sayımı

Görüntü İşleme dersi ara sınav çalışması  
Öğrenci numarası: **264309500**  
Rastgele seçim seed'i: **11526** (`09500 + 2026`)

## Çalışmanın konusu

Bu projede Fluo-N2DH-GOWT1 veri setindeki hücreleri klasik görüntü işleme yöntemleriyle saydım. İlk olarak Otsu eşikleme ve bağlı bileşen analizini başlangıç yöntemi olarak aldım. Sonra yoğunluk dönüşümleri, filtreler, eşikleme türleri ve morfolojik işlemleri değiştirerek 24 deney yaptım. En iyi üç adayı doğrulama karelerinde karşılaştırdım; seçilen ayarı ayrı bir sekans üzerinde test ettim.

Veri seti ve deney sonuçları kodda ayrı tutuluyor. Orijinal TIFF dosyaları ile anotasyonlar GitHub'a eklenmemiştir.

## Veri seti ve görüntü seçimi

- Veri seti: [Cell Tracking Challenge – 2D datasets](https://celltrackingchallenge.net/2d-datasets/)
- İndirilen arşiv: [Fluo-N2DH-GOWT1, training](https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip)
- İki sekans: `01` ve `02`; her biri 92 adet 1024×1024 görüntü içeriyor.
- Geliştirme (01): `002, 028, 037, 038, 042, 061, 070, 071, 076, 078`
- Doğrulama (01): `010, 013, 015, 048, 053`
- Test (02): `004–008`, `019–023`, `086–090` olmak üzere üç blok.

Seçim `selected_frames.csv` dosyasında kayıtlıdır. Geliştirme ve doğrulama kareleri çakışmaz, aralarında en az beş ara kare bulunur. Aynı boşluk koşulu test blokları için de uygulanmıştır. Resmî challenge test arşivi kullanılmamıştır; bütün görüntüler training arşivinden alınmıştır.

## Denediğim yöntemler

Başlangıç yöntemi: gri görüntü → Otsu → bağlı bileşenler → 40 pikselden küçük bölgeleri eleme.

Karşılaştırmalarda yüzde 2–98 kontrast germe, CLAHE ve histogram eşitleme; Gaussian 3×3/5×5, medyan 5×5 ve keskinleştirme; sabit, Otsu ve adaptif eşikleme; erosion, dilation, opening, closing ve watershed kullandım. Deneylerin ayarları ile geliştirme MAE sonuçları `experiments.csv` içinde bulunuyor.

Geliştirme kümesinde E16 (5×5 opening) MAE 2,70 ile ilk sıradaydı. Buna rağmen doğrulamada E15 (3×3 opening) daha iyi sonuç verdi. Bu nedenle şu zinciri seçtim:

`%2–98 kontrast germe → Gaussian 3×3 → Otsu → opening 3×3 → bağlı bileşen sayımı (min. alan 40)`

Seçim `final_pipeline.json` dosyasına yazılmıştır. Test sonuçları görüldükten sonra parametrelerde değişiklik yapılmamıştır.

## Ölçtüğüm sonuçlar

| Ölçüt | Başlangıç yöntemi | E15 |
|---|---:|---:|
| Test MAE | 3,467 | 13,667 |
| Test RMSE | 3,847 | 13,811 |
| Bias | -1,867 | +13,667 |
| Tam doğru sayım | %6,67 | %0,00 |
| Ortalama süre (s/kare) | 0,0076 | 0,0180 |

E15, 15 test karesinin tamamında fazla sayım yaptı. Bu sonuç başlangıç yönteminin söz konusu test sekansında daha başarılı olduğunu gösteriyor. Hatanın olası nedenlerini raporun tartışma bölümünde ele aldım; test görüntülerine bakarak yeni bir ayar seçmedim. Ayrıntılı sayımlar `test_results.csv`, özet metrikler `test_metrics.csv` içinde bulunuyor.

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `analysis_core.py` | Görüntü seçimi, filtreleme, segmentasyon, deneyler ve değerlendirme |
| `main.ipynb` | Colab/Jupyter üzerinden aşamalı çalıştırma |
| `requirements.txt` | Python bağımlılıkları |
| `selected_frames.csv` | Kullanılan 30 kare ve seed |
| `experiments.csv` | 24 deneyin yöntem ve performans bilgileri |
| `validation_results.csv` | Üç adayın doğrulama sonuçları |
| `final_pipeline.json` | Test öncesinde seçilen parametreler |
| `test_predictions_locked.csv` | Test etiketleri okunmadan elde edilen tahminler |
| `lock_manifest.json` | Kilitlenen dosyaların SHA256 özetleri |
| `test_results.csv`, `test_metrics.csv` | Test referanslarıyla yapılan karşılaştırma |
| `make_report.py` | Teknik raporu oluşturma |
| `verify_results.py` | Mevcut sonuç dosyalarının tutarlılık kontrolü |
| `SONUCLAR.md` | Deney bulgularının kısa değerlendirmesi |

Ayrıca `development_per_frame.csv`, `validation_per_frame.csv`, `dataset_diagnostics.csv` ve `analysis_environment.json` ayrıntılı inceleme ve tekrarlanabilirlik için eklenmiştir.

## Yeniden çalıştırma

Yerel Jupyter gerekmiyor. Projede `main.ipynb` bulunduğu için Google Colab'da çalıştırılabilir; alternatif olarak Python komut satırı kullanılabilir. Kod yeniden çalıştırılacaksa mevcut kilitlenmiş sonuçların bulunduğu klasör yerine temiz bir proje klasörü açılmalıdır. Notebook, daha önce üretilmiş kilit dosyalarını gördüğünde bunu ayrı bir yeniden çalışma klasöründe yapar.

1. Eğitim arşivini indirin ve `data/Fluo-N2DH-GOWT1/` altına açın. `01`, `02`, `01_GT`, `02_GT` klasörleri bu dizinin altında olmalıdır.
2. `pip install -r requirements.txt` komutuyla paketleri yükleyin.
3. Temiz klasörde `python analysis_core.py 1 --data data/Fluo-N2DH-GOWT1` komutunu çalıştırın. Bu aşama 24 deneyi, doğrulamayı ve referans okumadan test tahminlerini kaydeder.
4. Birinci aşamanın dosyalarını ayrı bir Git commit'i ile kaydedin. GitHub'a gönderip commit kimliğini not edin.
5. `python analysis_core.py 2 --data data/Fluo-N2DH-GOWT1 --lock-commit COMMIT_SHA` komutunu çalıştırın. İkinci aşama önce commit'teki kilit dosyalarını doğrular, sonra test referanslarını okur.
6. `python make_report.py` ve `python verify_results.py` komutlarını çalıştırın. Test sonuçlarını ikinci bir Git commit'ine alın.

GitHub üzerinde `01442c5` commit'iyle kilitli tahminler, ardından `f1f7afc` commit'iyle bağımsız test değerlendirmesi yüklenmiştir. Bu kayıtlar dosyaların GitHub'a yüklenme sırasını gösterir; test referanslarının ilk kez ne zaman okunduğunu tek başına kanıtlamaz. Deney sırasında oluşturulan özgün yerel Git geçmişi ayrı bir kayıt olarak korunmuştur.

## Veri paylaşımı

Ödevdeki veri seti görüntülerini ve TRA/SEG maskelerini GitHub üzerinde yeniden dağıtmıyorum. Orijinal görüntü içeren şekiller ve teknik PDF üniversitenin özel teslim ortamı için ayrılmıştır. `private_figures/` ile `data/` `.gitignore` kapsamındadır. Notebook'un GitHub sürümünde gömülü mikroskop çıktıları bulunmamalıdır.
