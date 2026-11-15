"""Deney sonuçlarını CSV dosyalarından okuyarak teknik raporu hazırlar.
Kaynakça ana metinden ayrı bir sayfada yer alır.
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether, HRFlowable)

REPORT_NAME='Goruntu_Isleme_Rapor_264309500.pdf'


def _fonts():
    opts=[('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
           '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
          ('/usr/local/share/fonts/DejaVuSans.ttf','/usr/local/share/fonts/DejaVuSans-Bold.ttf')]
    for r,b in opts:
        if Path(r).exists() and Path(b).exists():
            pdfmetrics.registerFont(TTFont('TR',r))
            pdfmetrics.registerFont(TTFont('TRB',b))
            return 'TR','TRB'
    raise RuntimeError('Türkçe karakterler için DejaVuSans fontu bulunamadı. Colab ortamında fonts-dejavu-core kurun.')


def build(workdir='.'):
    wd=Path(workdir).resolve()
    required=['test_results.csv','test_metrics.csv','experiments.csv','validation_results.csv',
              'selected_frames.csv','final_pipeline.json','dataset_diagnostics.csv']
    for filename in required:
        if not (wd/filename).is_file():
            raise FileNotFoundError(f'{filename} eksik: Testi tamamlayıp kilit commit doğrulaması yapın.')
    selected=pd.read_csv(wd/'selected_frames.csv',dtype={'sequence':str})
    exp=pd.read_csv(wd/'experiments.csv')
    val=pd.read_csv(wd/'validation_results.csv')
    tests=pd.read_csv(wd/'test_results.csv')
    metrics=pd.read_csv(wd/'test_metrics.csv')
    diag=pd.read_csv(wd/'dataset_diagnostics.csv')
    pipe=json.loads((wd/'final_pipeline.json').read_text(encoding='utf8'))
    figs=wd/'private_figures'
    normal,bold=_fonts()
    ink=colors.HexColor('#173849');slate=colors.HexColor('#526675');light=colors.HexColor('#eaf2f5')
    styles={
      'title':ParagraphStyle('t',fontName=bold,fontSize=18,leading=24,textColor=ink,spaceAfter=14),
      'h':ParagraphStyle('h',fontName=bold,fontSize=11.8,leading=16,textColor=ink,spaceBefore=8,spaceAfter=6),
      'sh':ParagraphStyle('sh',fontName=bold,fontSize=9.5,leading=13,textColor=ink,spaceBefore=8,spaceAfter=4),
      'p':ParagraphStyle('p',fontName=normal,fontSize=8.8,leading=13,spaceAfter=6,textColor=colors.HexColor('#263640')),
      'small':ParagraphStyle('s',fontName=normal,fontSize=7.3,leading=10,spaceAfter=4,textColor=slate),
      'caption':ParagraphStyle('c',fontName=normal,fontSize=7.7,leading=11,spaceAfter=9,textColor=slate),
      'number':ParagraphStyle('n',fontName=normal,fontSize=7,leading=9),
      'tablehead':ParagraphStyle('th',fontName=bold,fontSize=7,leading=9,textColor=colors.white),
    }
    P=lambda text,style='p':Paragraph(str(text).replace('&','&amp;'),styles[style])
    story=[]
    def put(text,kind='p'): story.append(P(text,kind))
    def h(text):put(text,'h')
    def small(text):put(text,'small')
    def nxt():story.append(PageBreak())
    def tbl(head,rows,widths=None,fontsize=7.0):
        data=[[P(str(x),'tablehead') for x in head]]+[[P(str(x),'number') for x in row] for row in rows]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([
          ('BACKGROUND',(0,0),(-1,0),ink),('TEXTCOLOR',(0,0),(-1,0),colors.white),
          ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),
          ('FONTNAME',(0,0),(-1,-1),normal),('FONTSIZE',(0,0),(-1,-1),fontsize),
          ('BOTTOMPADDING',(0,0),(-1,0),6),('TOPPADDING',(0,0),(-1,0),6),
          ('BOTTOMPADDING',(0,1),(-1,-1),3),('TOPPADDING',(0,1),(-1,-1),3),
          ('GRID',(0,0),(-1,-1),.22,colors.HexColor('#c6d5dc')),
          ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ]))
        story.append(t);story.append(Spacer(1,5))
    def fig(file, caption, width_mm=173, maxh_mm=105):
        path=figs/file
        if not path.exists():raise FileNotFoundError(f'Zorunlu şekil yok: {path}')
        from PIL import Image as PILImage
        with PILImage.open(path) as pil: w,h=pil.size
        desired_w=width_mm*mm
        desired_h=desired_w*h/w
        if desired_h>maxh_mm*mm:
            factor=(maxh_mm*mm)/desired_h
            desired_h*=factor;desired_w*=factor
        story.append(Image(str(path),width=desired_w,height=desired_h,hAlign='CENTER'))
        put(caption,'caption')

    def pagetext(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#d8e4e9'))
        canvas.line(19*mm,20*mm,191*mm,20*mm)
        canvas.setFont(normal,7.5);canvas.setFillColor(slate)
        canvas.drawString(19*mm,15.4*mm,'Görüntü İşleme - Ara Sınav Çalışması')
        canvas.drawRightString(191*mm,15.4*mm,f'{doc.page}')

    doc=SimpleDocTemplate(str(wd/REPORT_NAME),pagesize=A4,
                          rightMargin=18*mm,leftMargin=18*mm,
                          topMargin=19*mm,bottomMargin=24*mm,
                          title='Klasik Yöntemlerle Mikroskobik Hücre Sayımı',author='264309500')
    usable_w=174*mm

    # PAGE 1
    story.append(Spacer(1,15*mm))
    put('KLASİK GÖRÜNTÜ İŞLEME YÖNTEMLERİYLE MİKROSKOBİK HÜCRE SAYIMI','title')
    put('Yüksek Lisans / Görüntü İşleme - Ara Sınav Çalışması','h')
    put('Öğrenci Numarası: <b>264309500</b> | Tekrarlanabilir seed: <b>11526</b>','p')
    h('Özet')
    best=pipe['config'];base=metrics.set_index('method').loc['Baseline'];fin=metrics.set_index('method').loc['Nihai']
    put('Fluo-N2DH-GOWT1 veri setindeki hücreleri klasik görüntü işleme adımlarıyla saymayı '
        've bu adımların sayım hatasını nasıl değiştirdiğini inceledim. Sequence 01 üzerinde 24 ayrı '
        'ayar denedim; bunlardan üçü doğrulama aşamasına geçti. Doğrulama sonuçlarına göre '
        'kontrast germe, Gaussian 3×3, Otsu ve 3×3 opening içeren E15 yöntemi seçildi. '
        'Sequence 02 üzerinden yaptığım 15 karelik bağımsız testte ise E15 yönteminin '
        f'MAE değeri {fin.MAE:.2f}, başlangıç yönteminin {base.MAE:.2f} çıktı. '
        'Bu fark, doğrulamada iyi görünen bir ayarın başka bir sekansa her zaman aktarılamadığını gösterdi.')
    h('1. Giriş ve Problem Tanımı')
    put('Floresan mikroskop görüntülerinde hücre çekirdekleri arka plandan daha parlak görünse de '
        'hepsinin parlaklığı ve büyüklüğü aynı değildir. Üstelik bazı hücreler birbirine çok yakındır. '
        'Bu nedenle yalnızca eşikleme uygulamak yeterli olmayabilir: kimi zaman tek hücre birkaç '
        'parçaya ayrılır, kimi zaman iki hücre tek bir nesne gibi sayılır.')
    put('Çalışmada asıl sorum, hangi işlemlerin hücre sayısını daha güvenilir hâle getirdiğiydi. '
        'Önce Otsu eşikleme ve bağlı bileşen analizinden oluşan basit bir yöntem kurdum. '
        'Daha sonra kontrast, filtre, eşik ve morfoloji seçeneklerini karşılaştırıp seçtiğim '
        'yöntemi ayrı bir görüntü sekansında sınadım.')
    h('Veri Ayrımı')
    put('Geliştirme ve doğrulama kareleri 01 sekansından, test kareleri 02 sekansından seçildi. '
        'Test etiketleri yöntem seçiminde kullanılmadı. Nihai ayarlar ve test tahminleri '
        'Git geçmişine kaydedildikten sonra referans hücre sayıları okunarak değerlendirme yapıldı. '
        'Ödevin kapsamına uygun olarak derin öğrenme modeli kullanılmadı.')
    small('Referans sayı, her karedeki takip etiketlerinde sıfır dışındaki farklı kimliklerin sayılmasıyla bulundu. '
          'Burada değerlendirilen başarı piksel düzeyindeki maske örtüşmesi değil, hücre sayımıdır.')
    nxt()

    # PAGE 2
    h('2. Veri Seti ve Deneysel Tasarım')
    put('Veri olarak Cell Tracking Challenge içindeki Fluo-N2DH-GOWT1 eğitim arşivini kullandım. '
        '01 ve 02 isimli iki sekansın her birinde 92 adet, 1024×1024 boyutunda gri seviyeli görüntü var. '
        'Kareler beşer dakikalık aralıklarla kaydedilmiş. Görüntüde geniş bir karanlık arka planın '
        'bulunması histogramın düşük yoğunluk değerlerinde toplanmasına yol açıyor.')
    put('Seçimin tekrar üretilebilmesi için öğrenci numaramın son beş hanesi ile 2026 toplamını '
        'seed olarak kullandım: 9500 + 2026 = 11526. 01 sekansından önce 5 doğrulama, '
        'ardından bu karelerden en az 5 ara kare uzaklıkta 10 geliştirme görüntüsü seçtim. '
        '02 sekansındaki test kümesi üç ayrı beş karelik bloktan oluşuyor.')
    rows=[(str(r.group),str(r.sequence),f'{int(r.frame):03d}',str(int(r.block)),str(r.filename))
          for r in selected.itertuples()]
    tbl(['Grup','Sekans','Kare','Blok','Dosya'],rows,[31*mm,20*mm,17*mm,16*mm,90*mm])
    put('Tablo 1. Rastgele seçim sonucunda kullanılan 30 görüntü.','caption')
    put(f"Geliştirme grubunun piksel yoğunluğu ortalaması {diag['mean'].mean():.2f}; "
        f"en düşük ve en yüksek piksel değerleri ise {int(diag['min'].min())} ile "
        f"{int(diag['max'].max())} arasındadır. Bu ölçüler yalnızca geliştirme karelerinden alınmıştır.", 'small')
    nxt()

    # PAGE 3
    h('3. Yöntem ve Pipeline Tasarımı')
    put('İlk karşılaştırma için gri görüntüye Otsu eşikleme uygulayıp bağlı bileşenleri saydım. '
        f'{pipe["baseline"]["min_area"]} pikselden küçük bölgeleri dışarıda bıraktım. '
        'Bu başlangıç yönteminin ayarları sonraki testlerde değiştirilmedi.')
    put('Diğer denemelerde orijinal görüntüye ek olarak yüzde 2–98 aralığında kontrast germe, '
        'histogram eşitleme ve CLAHE kullandım (clipLimit=2, 8×8 karo). '
        'Gürültü ve hücre sınırları için Gaussian 3×3/5×5, medyan 5×5 ve bir keskinleştirme '
        'çekirdeğini karşılaştırdım. Eşiklemede sabit 85/120, Otsu ve adaptif yöntemleri denedim.')
    put('İkili maskelere 3×3 ve 5×5 eliptik elemanlarla erosion, dilation, opening, closing '
        'uyguladım; opening–closing sırasını da tersine çevirdim. Yakın hücreleri ayırmak için '
        'distance transform ile watershed seçeneğini ekledim. Nesne sayısını hesaplarken '
        'en küçük alan sınırını tüm ilgili karşılaştırmalarda sabit tuttum.')
    fig('sekil1_akis_diyagrami.png','Şekil 1. Testten önce seçilen E15 işlem adımları.',170,45)
    h('Seçilen Yöntemin Ayarları')
    tbl(['Adım','Seçilen değer'],[
        ('Deney',best['id']),('Yoğunluk',best['intensity']),('Filtre',best['filter']),
        ('Segmentasyon',best['segment']),('Morfoloji',str(best['morph'])),
        ('En küçük alan',str(best['min_area'])),
        ('Watershed',str(best['watershed'])),('Min mesafe',str(best['min_distance']))],
        [50*mm,124*mm])
    put('Bağlı bileşen sayısı, her zaman gerçek hücre sayısıyla aynı değildir. Özellikle '
        'temas eden hücreler ve parçalı maskeler bu iki sayının ayrışmasına neden olabilir.')
    nxt()

    # PAGE 4
    h('4. Kontrollü Deneyler ve Parametre Analizi')
    put('Yöntemleri 10 geliştirme karesinin tamamında çalıştırdım. Her denemede hangi '
        'adımın değiştiğini izleyebilmek için temel işlem zincirini korudum; farklı '
        'filtreleri, eşikleri ve morfolojik sıraları bu zincir üzerinde karşılaştırdım.')
    exp_rows=[]
    for r in exp.sort_values('experiment_id').itertuples():
        ops=json.loads(r.morphology)
        morph_text=' + '.join(f'{name}{size}' for name,size in ops) if ops else '-'
        if r.watershed:
            morph_text=f'WS{int(r.min_distance)} + '+morph_text
        exp_rows.append((r.experiment_id.split('_')[0],r.intensity,r.filter,r.segmentation,
                         morph_text,f'{r.development_MAE:.2f}',f'{r.mean_seconds:.3f}'))
    tbl(['ID','Yoğunluk','Filtre','Eşik','Morfoloji','MAE','s/kare'],exp_rows,
        [13*mm,29*mm,22*mm,26*mm,46*mm,18*mm,20*mm])
    put('Tablo 2. Yirmi dört deneyin geliştirme hataları ve ortalama işlem süreleri.','caption')
    top=exp.sort_values(['development_MAE','development_RMSE']).head(1).iloc[0]
    put(f'Bu kümede en düşük hata {top.experiment_id} için {top.development_MAE:.2f} bulundu. '
        'Ancak bu tek başına yöntemi seçmek için yeterli değildi; üç adayın '
        'doğrulama sonuçlarını ayrıca karşılaştırdım.')
    put('Kontrast germe tek başına yarar sağlamadı: ham Otsu kullanan E01 için MAE 3,90 iken '
        'E02 için 4,60 oldu. Gaussian çekirdeğini 3×3 yerine 5×5 seçmek ise aynı ayarlarda '
        'hatayı 4,60’tan 4,40’a indirdi. Keskinleştirme kullanılan E08’de hata 8,30’a çıktı.', 'small')
    put('Morfolojik opening daha belirgin fark yarattı. Morfoloji uygulanmayan E05’in '
        'MAE değeri 4,60; 3×3 opening kullanan E15’in 3,40 ve 5×5 opening kullanan '
        'E16’nın 2,70 oldu. İşlem sırası da etkiliydi: opening–closing (E18) 3,70, '
        'closing–opening (E19) 4,40 hata verdi.', 'small')
    put('Bazı yöntemler bu görüntülerde oldukça fazla nesne üretti. Örneğin histogram '
        'eşitlemeli E04’te MAE 670,50; adaptif eşiklemenin E11 ve E12 ayarlarında '
        '804,60 ve 984,30 bulundu. Bu nedenle bütün yöntemler benzer ölçüde '
        'başarılı kabul edilmedi.', 'small')
    nxt()

    # PAGE 5
    h('4.1. Görüntüler Üzerinden Karşılaştırma')
    fig('sekil2_yogunluk_ve_histogram.png',
        'Şekil 2. Aynı geliştirme karesine uygulanan yoğunluk dönüşümleri ve histogramları.',174,102)
    fig('sekil3_segmentasyon_morfoloji.png',
        'Şekil 3. Aynı karede eşikleme ve morfolojik işlemlerden elde edilen maskeler.',164,100)
    nxt()

    # PAGE 6
    h('5. Doğrulama ve Nihai Yöntemin Seçilmesi')
    put('Geliştirmedeki en düşük üç MAE değerini veren E16, E15 ve E10 ayarlarını '
        'beş doğrulama karesinde denedim. Karar ölçütü önce MAE, eşitlik hâlinde '
        'RMSE, süre ve işlem karmaşıklığıydı.')
    rows=[(r.pipeline,f'{r.validation_MAE:.3f}',f'{r.validation_RMSE:.3f}',
           f'{r.mean_seconds:.3f}','Seçildi' if r.selected else '—') for r in val.itertuples()]
    tbl(['Pipeline','MAE','RMSE','s/kare','Seçim'],rows,[72*mm,26*mm,26*mm,26*mm,24*mm])
    put('Tablo 3. Üç adayın beş doğrulama karesindeki sonuçları.','caption')
    chosen=val[val.selected].iloc[0]
    put('Geliştirmede E16 daha iyi görünüyordu (MAE 2,70). Buna karşılık doğrulamada '
        '3×3 opening kullanan E15 daha düşük hata verdi: E15 için 4,20; E16 için 4,80 '
        've E10 için 5,60. Bu yüzden E15’i seçip parametrelerini testten önce sabitledim.')
    fig('sekil4_pipeline_adimlari.png',
        'Şekil 4. E15 yönteminde orijinal görüntüden son maskeye kadar işlem aşamaları.',174,73)
    h('Test Tahminlerinin Sabitlenmesi')
    put('Seçilen parametreleri final_pipeline.json dosyasına, 15 test tahminini de '
        'test_predictions_locked.csv dosyasına yazdım. Her iki dosyanın SHA256 özeti '
        'lock_manifest.json içinde saklandı. İlk Git commit’i bu dosyaları içeriyor. '
        'Test etiketlerini okuyan ikinci aşama, kayıtlı dosyaların aynı olduğunu '
        'kontrol ettikten sonra çalışıyor.')
    nxt()

    # PAGE 7
    h('6. Bağımsız Test Sonuçları')
    put('Sequence 02’den seçilen 15 görüntüye ait tahminleri sabitledikten sonra '
        'referans etiketlerini okudum. Tablo 4, kare bazındaki gerçek sayıları ve '
        'iki yöntemin hatalarını yan yana gösteriyor.')
    rows=[(f'{int(r.frame):03d}',str(int(r.reference_count)),str(int(r.baseline_prediction)),
           str(int(r.final_prediction)),str(int(r.baseline_absolute_error)),str(int(r.final_absolute_error)))
          for r in tests.itertuples()]
    tbl(['Kare','Referans','Baseline','Nihai','|Hata| B','|Hata| N'],rows,
        [24*mm,30*mm,30*mm,30*mm,30*mm,30*mm])
    put('Tablo 4. Test görüntülerindeki hücre sayıları ve mutlak hatalar.','caption')
    rows=[]
    for _,r in metrics.iterrows():
        rows.append((r['method'],f"{r['MAE']:.3f}",f"{r['RMSE']:.3f}",f"{r['Bias']:.3f}",
                     f"{100*r['Exact-count Accuracy']:.1f}%",f"{r['mean_seconds']:.3f}"))
    tbl(['Yöntem','MAE','RMSE','Bias','Tam isabet','s/kare'],rows,[44*mm,23*mm,23*mm,23*mm,35*mm,26*mm])
    put('Tablo 5. Test kümesindeki toplu sonuçlar (Bias = tahmin − referans ortalaması).','caption')
    put(f'Testte başlangıç yönteminin MAE değeri {base.MAE:.2f}, E15’in ise {fin.MAE:.2f} '
        'oldu. E15’in hatası daha yüksek olduğu gibi işlem süresi de daha uzundu '
        f'({fin.mean_seconds:.3f} s/kare; başlangıç {base.mean_seconds:.3f} s/kare). '
        'Dolayısıyla doğrulamada tercih edilen yöntem, bağımsız sekans için '
        'başlangıç yöntemini geçemedi.')
    positive=int((tests.final_signed_error>0).sum())
    put(f'E15 bütün test karelerinde fazla sayım yaptı ({positive}/15). Referans sayı '
        f'{int(tests.reference_count.min())}–{int(tests.reference_count.max())} arasında değişirken '
        f'E15 tahminleri {int(tests.final_prediction.min())}–{int(tests.final_prediction.max())} '
        'aralığında kaldı. Örneğin 022. karede referans 24 ve başlangıç tahmini 24 iken '
        'E15 tahmini 40 oldu. Sorun tek bir görüntüye özgü görünmüyor.', 'small')
    nxt()

    # PAGE 8
    h('7. Zamansal Analiz, Tartışma ve Sonuç')
    fig('sekil5_zamansal_uc_blok.png',
        'Şekil 5. Üç beş karelik blokta referans, başlangıç ve E15 sayımlarının değişimi.',174,67)
    fig('sekil6_ornek_basari_hata.png',
        'Şekil 6. E15 için en düşük ve en yüksek hata ile yoğun hücreli bir kare. Altta ikili maskeler var.',156,90)
    block_stats=tests.groupby('block').agg(baseline_mae=('baseline_absolute_error','mean'),
                                           final_mae=('final_absolute_error','mean'))
    put('Üç blokta da E15 tahminleri referans sayısının üstünde kaldı. Blokların MAE '
        'değerleri sırasıyla '
        + ', '.join(f'{int(k)}. blokta {block_stats.loc[k,"final_mae"]:.2f}' for k in sorted(block_stats.index))
        + '. Dolayısıyla hata yalnızca birkaç sorunlu görüntüden kaynaklanmıyor. '
        'Grafiklerde başlangıç yönteminin referans eğrisine daha yakın seyrettiği görülüyor.')
    put('Şekil 6’daki maskelerde bazı hücrelerin içinde küçük ve birbirinden ayrı parlak '
        'alanlar seçiliyor. Kontrast germe ile eşikleme bu parçaları ayrı bileşenlere '
        'dönüştürüyor olabilir. Bu, fazla sayımı açıklayabilecek bir olasılık; '
        'hücre bazında eşleştirme yapmadığım için kesin neden olarak sunmuyorum.', 'small')
    put('Bu çalışmada 5×5 opening geliştirme görüntülerinde, 3×3 opening ise doğrulama '
        'görüntülerinde daha başarılı oldu. Fakat testte her iki aşamadan çıkarılan '
        'genel beklenti karşılanmadı. Özellikle iki sekans arasındaki görünüm farkı '
        've az sayıdaki doğrulama görüntüsü bu sonucun olası nedenleri arasında.', 'small')
    put('Sonuçta, karmaşık bir işlem zincirinin her zaman daha iyi sayım yapmadığını '
        'gördüm. Bu veri ayrımında basit Otsu yöntemi testte daha başarılıydı. '
        'İleride farklı sekansları kapsayan geliştirme verisi, daha ayrıntılı '
        'nesne birleştirme/ayırma kontrolü ve bağımsız bir dış test kümesiyle '
        'aynı soruyu yeniden incelemek yararlı olur. Mevcut test sonucuna göre '
        'E15’in parametrelerini değiştirmedim.')
    nxt()

    # PAGE 9 - References (excluded from 6-8 page limit)
    h('Kaynakça (sayfa sınırı dışındadır)')
    for entry in [
      'Cell Tracking Challenge. 2D+Time Datasets. https://celltrackingchallenge.net/2d-datasets/',
      'Ulman, V. ve diğerleri (2017). An objective comparison of cell-tracking algorithms. Nature Methods, 14, 1141–1152.',
      'Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66.',
      'Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization. Graphics Gems IV.',
      'Vincent, L. ve Soille, P. (1991). Watersheds in Digital Spaces: An Efficient Algorithm Based on Immersion Simulations. IEEE TPAMI, 13(6), 583–598.',
      'OpenCV Documentation. https://docs.opencv.org/',
      'scikit-image Documentation. https://scikit-image.org/',
    ]: put(entry)
    small('Tekrarlanabilirlik: selected_frames.csv, experiments.csv, validation_results.csv, '
          'final_pipeline.json, test_predictions_locked.csv ve test_results.csv ile '
          'analysis_environment.json dosyaları referans alınmalıdır.')
    doc.build(story,onFirstPage=pagetext,onLaterPages=pagetext)
    return wd/REPORT_NAME


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--workdir',default='.')
    args=parser.parse_args()
    print(build(args.workdir))
