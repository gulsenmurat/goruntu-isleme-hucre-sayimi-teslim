"""Görüntü İşleme yüksek lisans ara sınavı - deney/analiz çekirdeği.

Yalnızca klasik görüntü işleme yöntemleri kullanılır.
Birinci aşama 02_GT etiketlerini okumaz. İkinci aşama test referanslarını
ancak kilit commit doğrulandıktan sonra kullanır.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import skimage
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import tifffile
from PIL import Image as PILImage

STUDENT_NO = '264309500'
SEED = int(STUDENT_NO[-5:]) + 2026
DATASET_URL = 'https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip'
DATASET_NAME = 'Fluo-N2DH-GOWT1'
BASELINE_MIN_AREA = 40  # Geliştirme öncesinde sabitlenmiş örnek alan filtresi (piksel)
FRAME_REGEX = re.compile(r'^t(\d+)\.(?:tif|tiff)$', re.I)


@dataclass(frozen=True)
class Dataset:
    root: Path
    frames01: dict[int, Path]
    frames02: dict[int, Path]

    def image(self, seq: str, frame: int) -> np.ndarray:
        path = (self.frames01 if seq == '01' else self.frames02)[int(frame)]
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            img = tifffile.imread(path)
        if img.ndim != 2:
            raise ValueError(f'Tek kanallı görüntü bekleniyor: {path}; boyut {img.shape}')
        return np.ascontiguousarray(img)

    def reference(self, seq: str, frame: int) -> int:
        """Yalnızca izin verilen aşama tarafından çağrılır. Arka plan etiketi 0 dışlanır."""
        if seq not in ('01', '02'):
            raise ValueError('Bilinmeyen sekans')
        path = self.root / f'{seq}_GT' / 'TRA' / f'man_track{int(frame):03d}.tif'
        if not path.is_file():
            raise FileNotFoundError(f'Referans maske yok: {path}')
        # Challenge TRA masks can use LZW-compressed TIFF, for which tifffile
        # requires an optional imagecodecs wheel. Pillow reads these reliably.
        with PILImage.open(path) as image:
            mask = np.asarray(image)
        return int(np.count_nonzero(np.unique(mask)))


def discover_dataset(root: str | Path) -> Dataset:
    path = Path(root).expanduser().resolve()
    if not (path / '01').is_dir() or not (path / '02').is_dir():
        nested = path / DATASET_NAME
        if (nested / '01').is_dir():
            path = nested
        else:
            raise FileNotFoundError(f'01 ve 02 görüntü dizinleri bulunamadı: {path}')

    def collect(seq: str) -> dict[int, Path]:
        result = {}
        for file in (path / seq).iterdir():
            m = FRAME_REGEX.fullmatch(file.name)
            if m:
                index = int(m.group(1))
                if index in result:
                    raise ValueError(f'Mükerrer kare numarası: {file}')
                result[index] = file
        return dict(sorted(result.items()))

    a, b = collect('01'), collect('02')
    if not a or not b:
        raise RuntimeError('Sekanslardan görüntüler okunamadı.')
    return Dataset(path, a, b)


def _draw_selection(ids01, ids02):
    """Seed ile deterministik seçim; dev-valid arası 5 ARA kare şartı sağlanır."""
    rng = np.random.default_rng(SEED)
    ids01 = sorted(int(x) for x in ids01)
    ids02 = sorted(int(x) for x in ids02)
    if len(ids01) < 30 or len(ids02) < 30:
        raise ValueError('Sekans kare sayısı beklenenden düşük.')

    # Validation önce; uygun development örnekleri onun +-5 komşuluğundan dışlanır.
    for _ in range(10_000):
        validation = sorted(map(int, rng.choice(ids01, size=5, replace=False)))
        eligible = [x for x in ids01 if all(abs(x - y) >= 6 for y in validation)]
        if len(eligible) >= 10:
            development = sorted(map(int, rng.choice(eligible, size=10, replace=False)))
            break
    else:
        raise RuntimeError('Seed ile geçerli 01 split üretilemedi.')

    # Bir test bloğu, veride gerçekten var olan tam 5 ardışık karenin başlangıcıdır.
    idset = set(ids02)
    starts_ok = [x for x in ids02 if all(x + d in idset for d in range(5))]
    for _ in range(100_000):
        starts = sorted(map(int, rng.choice(starts_ok, size=3, replace=False)))
        # İlk blok [s,s+4]; arada en az 5 kare = sonraki başlangıç >= önceki+10.
        if all(starts[j + 1] - (starts[j] + 4) >= 6 for j in range(2)):
            break
    else:
        raise RuntimeError('Çakışmayan 3 test bloğu bulunamadı.')

    rows = []
    for x in development:
        rows.append(dict(group='development', sequence='01', frame=x, filename=f'01/t{x:03d}.tif', block=0, seed=SEED))
    for x in validation:
        rows.append(dict(group='validation', sequence='01', frame=x, filename=f'01/t{x:03d}.tif', block=0, seed=SEED))
    for block_no, start in enumerate(starts, 1):
        for x in range(start, start + 5):
            rows.append(dict(group='test', sequence='02', frame=x, filename=f'02/t{x:03d}.tif', block=block_no, seed=SEED))
    result = pd.DataFrame(rows)
    assert len(result) == 30
    return result


def frame_selection(dataset: Dataset, workdir: Path) -> pd.DataFrame:
    selected = _draw_selection(dataset.frames01.keys(), dataset.frames02.keys())
    target = workdir / 'selected_frames.csv'
    if target.exists():
        old = pd.read_csv(target, dtype={'sequence': str, 'filename': str})
        pd.testing.assert_frame_equal(old, selected, check_dtype=False)
    else:
        selected.to_csv(target, index=False)
    if any(not (dataset.root / p).is_file() for p in selected['filename']):
        raise FileNotFoundError('selected_frames.csv içindeki en az bir görüntü bulunamadı')
    return selected


def to_u8(img: np.ndarray, mode: str) -> np.ndarray:
    img = img.astype(np.float32)
    if mode == 'original':
        # 16-bit kamera verisinin min-max aralıkta 8-bit gösterimi.
        lo, hi = float(img.min()), float(img.max())
    elif mode in ('stretch', 'clahe', 'equalize'):
        lo, hi = map(float, np.percentile(img, [2, 98]))
    else:
        raise ValueError(mode)
    norm = np.clip((img - lo) * (255.0 / max(hi - lo, 1e-9)), 0, 255).astype(np.uint8)
    if mode == 'clahe':
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(norm)
    if mode == 'equalize':
        return cv2.equalizeHist(norm)
    return norm


def apply_filter(gray: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'none': return gray
    if mode == 'gauss3': return cv2.GaussianBlur(gray, (3, 3), 0)
    if mode == 'gauss5': return cv2.GaussianBlur(gray, (5, 5), 0)
    if mode == 'median5': return cv2.medianBlur(gray, 5)
    if mode == 'sharpen':
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)
        return cv2.filter2D(gray, -1, kernel)
    raise ValueError(mode)


def segment(gray: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'otsu':
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0
    if mode == 'fixed85': return gray > 85
    if mode == 'fixed120': return gray > 120
    if mode == 'adaptive31':
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, -2) > 0
    if mode == 'adaptive51':
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 51, -2) > 0
    raise ValueError(mode)


def morph(binary: np.ndarray, steps: list[list[Any]]) -> np.ndarray:
    result = binary.astype(np.uint8) * 255
    for method, size in steps:
        elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(size), int(size)))
        if method == 'erode': result = cv2.erode(result, elem)
        elif method == 'dilate': result = cv2.dilate(result, elem)
        elif method == 'open': result = cv2.morphologyEx(result, cv2.MORPH_OPEN, elem)
        elif method == 'close': result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, elem)
        else: raise ValueError(method)
    return result > 0


def label_count(binary: np.ndarray, min_area: int, use_watershed: bool = False,
                min_distance: int = 9) -> tuple[int, np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    good = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= min_area) + 1
    keep = np.isin(labels, good)
    if not use_watershed:
        ids, objects = ndi.label(keep, structure=np.ones((3, 3), dtype=np.uint8))
        return int(objects), ids.astype(np.int32)

    distance = ndi.distance_transform_edt(keep)
    coords = peak_local_max(distance, min_distance=min_distance, labels=keep.astype(np.uint8),
                            exclude_border=False)
    markers = np.zeros(keep.shape, dtype=np.int32)
    for index, (y, x) in enumerate(coords, 1):
        markers[y, x] = index
    if not np.any(markers):
        return 0, markers
    split = watershed(-distance, markers=markers, mask=keep)
    unique, counts = np.unique(split[split > 0], return_counts=True)
    valid = unique[counts >= min_area]
    # lbl's positive IDs are cell instances, including disconnected watershed regions.
    # Vectorized relabeling preserves mask semantics while avoiding one full-image
    # scan per cell marker (important for 1024x1024 microscopy frames).
    lookup = np.zeros(int(split.max()) + 1, dtype=np.int32)
    lookup[valid] = np.arange(1, len(valid) + 1, dtype=np.int32)
    out = lookup[split]
    return int(len(valid)), out


def process(img: np.ndarray, config: dict, want_steps: bool = False):
    intensity = to_u8(img, config['intensity'])
    filtered = apply_filter(intensity, config['filter'])
    binary = segment(filtered, config['segment'])
    after_morph = morph(binary, config['morph'])
    count, labels = label_count(after_morph, config['min_area'],
                                config.get('watershed', False), config.get('min_distance', 9))
    if want_steps:
        return count, dict(original=img, intensity=intensity, filtered=filtered,
                           segment=binary.astype(np.uint8)*255,
                           morphology=after_morph.astype(np.uint8)*255, labels=labels)
    return count


def baseline(img: np.ndarray) -> int:
    """Ham gri seviye verisi + Otsu + bağlı bileşen + SABİT minimum alan."""
    if img.dtype not in (np.uint8, np.uint16):
        img = img.astype(np.uint16)
    peak = 65535 if img.dtype == np.uint16 else 255
    mask = cv2.threshold(img, 0, peak, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] > 0
    return label_count(mask, BASELINE_MIN_AREA)[0]


def config(id_: str, intensity='original', filter_='none', segment_='otsu',
           morph_=None, min_area=40, watershed_=False, min_distance=9):
    return dict(id=id_, intensity=intensity, filter=filter_, segment=segment_,
                morph=morph_ or [], min_area=min_area, watershed=watershed_,
                min_distance=min_distance)


def experiment_grid():
    """Sistematik varyasyonlar: tek parametre/işlem değişikliğiyle karşılaştırmalı deneyler."""
    return [
        config('E01_original_otsu'),
        config('E02_stretch_otsu', intensity='stretch'),
        config('E03_clahe_otsu', intensity='clahe'),
        config('E04_equalize_otsu', intensity='equalize'),
        config('E05_gauss3', intensity='stretch', filter_='gauss3'),
        config('E06_gauss5', intensity='stretch', filter_='gauss5'),
        config('E07_median5', intensity='stretch', filter_='median5'),
        config('E08_sharpen', intensity='stretch', filter_='sharpen'),
        config('E09_fixed85', intensity='stretch', filter_='gauss3', segment_='fixed85'),
        config('E10_fixed120', intensity='stretch', filter_='gauss3', segment_='fixed120'),
        config('E11_adaptive31', intensity='stretch', filter_='gauss3', segment_='adaptive31'),
        config('E12_adaptive51', intensity='stretch', filter_='gauss3', segment_='adaptive51'),
        config('E13_erode3', intensity='stretch', filter_='gauss3', morph_=[['erode',3]]),
        config('E14_dilate3', intensity='stretch', filter_='gauss3', morph_=[['dilate',3]]),
        config('E15_open3', intensity='stretch', filter_='gauss3', morph_=[['open',3]]),
        config('E16_open5', intensity='stretch', filter_='gauss3', morph_=[['open',5]]),
        config('E17_close3', intensity='stretch', filter_='gauss3', morph_=[['close',3]]),
        config('E18_open_close', intensity='stretch', filter_='gauss3', morph_=[['open',3],['close',3]]),
        config('E19_close_open', intensity='stretch', filter_='gauss3', morph_=[['close',3],['open',3]]),
        config('E20_watershed', intensity='stretch', filter_='gauss3', morph_=[['open',3]], watershed_=True),
        config('E21_watershed_d13', intensity='stretch', filter_='gauss3', morph_=[['open',3]], watershed_=True, min_distance=13),
        config('E22_clahe_open', intensity='clahe', filter_='gauss3', morph_=[['open',3]]),
        config('E23_clahe_watershed', intensity='clahe', filter_='gauss3', morph_=[['open',3]], watershed_=True),
        config('E24_median_adaptive', intensity='stretch', filter_='median5', segment_='adaptive51', morph_=[['open',3]]),
    ]


def scores(y_true, y_hat):
    y_true = np.asarray(y_true, dtype=float); y_hat = np.asarray(y_hat, dtype=float)
    err = y_hat-y_true
    return {'MAE': float(np.mean(np.abs(err))), 'RMSE': float(np.sqrt(np.mean(err**2))),
            'Bias': float(np.mean(err)), 'Exact-count Accuracy': float(np.mean(err == 0))}


def _savefig(path, fig):
    fig.savefig(path, dpi=145, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _show_img(ax, data, title):
    if np.issubdtype(data.dtype, np.integer):
        ax.imshow(data, cmap='gray')
    else:
        ax.imshow(data, cmap='gray')
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def baseline_visualization(img: np.ndarray, outdir: Path):
    """Ödevdeki baseline maskesi ve 8-komşuluklu tespit görselleştirmesi."""
    peak=65535 if img.dtype==np.uint16 else 255
    raw_mask=cv2.threshold(img,0,peak,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]>0
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(raw_mask.astype(np.uint8),8)
    good=[k for k in range(1,n) if int(stats[k,cv2.CC_STAT_AREA])>=BASELINE_MIN_AREA]
    preview=cv2.cvtColor(to_u8(img,'original'),cv2.COLOR_GRAY2RGB)
    for k in good:
        x,y,w,h,area=map(int,stats[k]);cx,cy=map(int,centroids[k])
        cv2.rectangle(preview,(x,y),(x+w,y+h),(255,105,40),1)
        cv2.putText(preview,str(k),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,0.3,(40,220,255),1)
    fig,axes=plt.subplots(1,3,figsize=(11,3.6))
    _show_img(axes[0],img,'Ham görüntü')
    _show_img(axes[1],raw_mask.astype(np.uint8),'Ham Otsu maskesi')
    axes[2].imshow(preview);axes[2].axis('off')
    axes[2].set_title(f'Bağlı bileşen tespitleri (n={len(good)})',fontsize=10)
    fig.tight_layout();_savefig(outdir/'baseline_tespit_gorseli.png',fig)


def generate_stage1_figures(img, chosen, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    names = [('original','Min-maks'), ('stretch','Yüzdelik 2-98'),
             ('clahe','CLAHE (clip=2)'), ('equalize','Histogram eşitleme')]
    fig, axes=plt.subplots(2,4,figsize=(14,6.8))
    for j,(name,title) in enumerate(names):
        data=to_u8(img,name)
        _show_img(axes[0,j], data, title)
        axes[1,j].hist(data.ravel(), bins=60, range=(0,255), color='#32768c')
        axes[1,j].set_xlabel('Yoğunluk (0-255)',fontsize=8)
        axes[1,j].set_ylabel('Piksel sayısı',fontsize=8)
        axes[1,j].tick_params(labelsize=7)
    fig.tight_layout(); _savefig(outdir/'sekil2_yogunluk_ve_histogram.png',fig)

    fig,axes=plt.subplots(2,3,figsize=(12,7.4))
    raw=to_u8(img,'stretch')
    methods=[('Sabit 85', segment(raw,'fixed85')),('Otsu',segment(raw,'otsu')),
             ('Adaptif 31',segment(raw,'adaptive31'))]
    for ax,(name,mask) in zip(axes[0],methods): _show_img(ax,mask.astype(np.uint8),name)
    masks=[('Erosion 3', morph(segment(raw,'otsu'),[['erode',3]])),
           ('Opening 3',morph(segment(raw,'otsu'),[['open',3]])),
           ('Closing 3',morph(segment(raw,'otsu'),[['close',3]]))]
    for ax,(name,mask) in zip(axes[1],masks): _show_img(ax,mask.astype(np.uint8),name)
    fig.tight_layout(); _savefig(outdir/'sekil3_segmentasyon_morfoloji.png',fig)

    count,steps=process(img, chosen, want_steps=True)
    fig,axes=plt.subplots(1,5,figsize=(15,3.7))
    for ax,key,title in zip(axes,('original','intensity','filtered','segment','morphology'),
                            ('Ham','Yoğunluk','Filtre','Eşik','Morfoloji')):
        _show_img(ax,steps[key],title)
    fig.suptitle(f"Nihai yöntem: {chosen['id']} | Tahmin: {count}",fontsize=11)
    fig.tight_layout(); _savefig(outdir/'sekil4_pipeline_adimlari.png',fig)

    fig,ax=plt.subplots(figsize=(12,2.2))
    ax.set_xlim(0,10);ax.set_ylim(0,1);ax.axis('off')
    labels=['Ham 16-bit','Yoğunluk','Filtre','Eşikleme','Morfoloji','Hücre sayısı']
    xx=np.linspace(0.75,9.25,len(labels))
    for j,(x,label) in enumerate(zip(xx,labels)):
        ax.text(x,0.52,label,ha='center',va='center',fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5',facecolor='#e8f2f6',edgecolor='#307e93'))
        if j<len(labels)-1: ax.annotate('',xy=(xx[j+1]-.64,.52),xytext=(x+.65,.52),arrowprops=dict(arrowstyle='->',lw=1.3))
    fig.tight_layout();_savefig(outdir/'sekil1_akis_diyagrami.png',fig)


def _filesha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metadata(ds, selected, workdir):
    rows=[]
    for _,item in selected[selected.group=='development'].iterrows():
        im=ds.image('01',int(item.frame))
        rows.append(dict(frame=int(item.frame),shape=f'{im.shape[0]}x{im.shape[1]}',
                         dtype=str(im.dtype),min=int(im.min()),max=int(im.max()),
                         mean=float(im.mean()),median=float(np.median(im)),
                         p2=float(np.percentile(im,2)),p98=float(np.percentile(im,98))))
    pd.DataFrame(rows).to_csv(workdir/'dataset_diagnostics.csv',index=False)
    environment = dict(student_number=STUDENT_NO,seed=SEED,dataset=DATASET_NAME,
                       python=sys.version.split()[0],numpy=np.__version__,opencv=cv2.__version__,
                       pandas=pd.__version__,scipy=scipy.__version__,skimage=skimage.__version__,
                       matplotlib=matplotlib.__version__,tifffile=tifffile.__version__,
                       baseline_min_area=BASELINE_MIN_AREA)
    (workdir/'analysis_environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2),encoding='utf-8')


def _details_row(config, result):
    return dict(experiment_id=config['id'],intensity=config['intensity'],
                filter=config['filter'],segmentation=config['segment'],
                morphology=json.dumps(config['morph'],ensure_ascii=False),
                min_area=config['min_area'],watershed=config['watershed'],
                min_distance=config['min_distance'],
                params_json=json.dumps(config,ensure_ascii=False,sort_keys=True),**result)


def run_stage1(root: str | Path = 'data/Fluo-N2DH-GOWT1', workdir: str | Path = '.'):
    """VERİ GELİŞTİRME + DOĞRULAMA -> TEST TAHMİNİNİ KİLİTLE; 02_GT ASLA OKUNMAZ."""
    workdir=Path(workdir).resolve()
    if (workdir/'test_predictions_locked.csv').exists() or (workdir/'test_results.csv').exists():
        raise RuntimeError('Kilitlenmiş test çıktısı mevcut: Aşama 1 yeniden çalıştırılamaz. Değerlendirilen dosyaları değiştirmeyin.')
    ds=discover_dataset(root)
    selection=frame_selection(ds,workdir)
    _metadata(ds,selection,workdir)
    dev=selection[selection.group=='development']
    val=selection[selection.group=='validation']
    test=selection[selection.group=='test']

    # DİKKAT: Yalnızca 01_GT maskelerinin okunması izinli.
    train_images={int(row.frame):ds.image('01',int(row.frame)) for _,row in dev.iterrows()}
    valid_images={int(row.frame):ds.image('01',int(row.frame)) for _,row in val.iterrows()}
    dev_reference={frame:ds.reference('01',frame) for frame in train_images}
    val_reference={frame:ds.reference('01',frame) for frame in valid_images}
    pd.DataFrame([dict(group='development',frame=x,reference_count=y) for x,y in dev_reference.items()] +
                 [dict(group='validation',frame=x,reference_count=y) for x,y in val_reference.items()]).to_csv(
                     workdir/'development_validation_reference.csv',index=False)

    result_rows=[];audit_rows=[]
    configurations=experiment_grid()
    for c in configurations:
        predicts=[]; references=[]; timings=[]
        for frame,img in train_images.items():
            t=time.perf_counter();p=process(img,c);elapsed=time.perf_counter()-t
            predicts.append(p);references.append(dev_reference[frame]);timings.append(elapsed)
            audit_rows.append(dict(experiment_id=c['id'],frame=frame,reference_count=dev_reference[frame],
                                   predicted_count=p,absolute_error=abs(p-dev_reference[frame]),seconds=elapsed))
        metrics=scores(references,predicts)
        result_rows.append(_details_row(c,dict(development_MAE=metrics['MAE'],development_RMSE=metrics['RMSE'],
                                               mean_seconds=float(np.mean(timings)))))
    experiments=pd.DataFrame(result_rows).sort_values(['development_MAE','development_RMSE','mean_seconds']).reset_index(drop=True)
    experiments.to_csv(workdir/'experiments.csv',index=False)
    pd.DataFrame(audit_rows).to_csv(workdir/'development_per_frame.csv',index=False)

    # Geliştirme en iyi üç ayrı aday. Doğrulamada seçim için kullanılacak üç pipeline.
    chosen_dev=experiments.head(3)['experiment_id'].tolist()
    candidates={c['id']:c for c in configurations if c['id'] in chosen_dev}
    validation_rows=[];val_audit=[]
    for id_ in chosen_dev:
        c=candidates[id_];pred=[];refs=[];timings=[]
        for frame,img in valid_images.items():
            start=time.perf_counter();count=process(img,c);elapsed=time.perf_counter()-start
            pred.append(count);refs.append(val_reference[frame]);timings.append(elapsed)
            val_audit.append(dict(pipeline=id_,frame=frame,reference_count=val_reference[frame],
                                  predicted_count=count,absolute_error=abs(count-val_reference[frame]),seconds=elapsed))
        m=scores(refs,pred)
        validation_rows.append(dict(pipeline=id_,validation_MAE=m['MAE'],validation_RMSE=m['RMSE'],
                                     mean_seconds=float(np.mean(timings)),
                                     complexity=len(c['morph'])+int(c['watershed'])+int(c['filter']!='none')))
    vr=pd.DataFrame(validation_rows).sort_values(['validation_MAE','validation_RMSE','mean_seconds','complexity']).reset_index(drop=True)
    winner=vr.iloc[0]['pipeline'];best=candidates[winner]
    vr['selected']=vr['pipeline'].eq(winner)
    vr.to_csv(workdir/'validation_results.csv',index=False)
    pd.DataFrame(val_audit).to_csv(workdir/'validation_per_frame.csv',index=False)

    pipeline=dict(student_number=STUDENT_NO,seed=SEED,dataset=DATASET_NAME,
                  baseline=dict(method='raw_Otsu_connected_components',min_area=BASELINE_MIN_AREA),
                  chosen_experiment=winner,config=best,selection_sha256=_filesha(workdir/'selected_frames.csv'),
                  selection_rule='5 validation + 10 development seq01 with >=5 intervening frames; 3x5 seq02 blocks with >=5 intervening frames',
                  validation_metrics=vr.iloc[0][['validation_MAE','validation_RMSE','mean_seconds']].to_dict(),
                  phase='STAGE1_LOCKED_BEFORE_TEST_REFERENCES')
    (workdir/'final_pipeline.json').write_text(json.dumps(pipeline,ensure_ascii=False,indent=2),encoding='utf-8')

    # Test görüntülerinin SADECE piksel verileri okunur, 02_GT/TRA yoluna erişilmez.
    locked=[]
    for _,item in test.iterrows():
        f=int(item.frame);img=ds.image('02',f)
        start=time.perf_counter();b=baseline(img);base_seconds=time.perf_counter()-start
        start=time.perf_counter();p=process(img,best);final_seconds=time.perf_counter()-start
        locked.append(dict(block=int(item.block),sequence='02',frame=f,filename=item.filename,
                           baseline_prediction=b,final_prediction=p,
                           baseline_seconds=base_seconds,final_seconds=final_seconds))
    pd.DataFrame(locked).to_csv(workdir/'test_predictions_locked.csv',index=False)
    (workdir/'lock_manifest.json').write_text(json.dumps({
        'student_number':STUDENT_NO,'seed':SEED,
        'selected_frames_sha256':_filesha(workdir/'selected_frames.csv'),
        'pipeline_sha256':_filesha(workdir/'final_pipeline.json'),
        'test_predictions_sha256':_filesha(workdir/'test_predictions_locked.csv'),
        'stage':'NO_TEST_REFERENCES_READ'
    },ensure_ascii=False,indent=2),encoding='utf-8')
    generate_stage1_figures(next(iter(train_images.values())),best,workdir/'private_figures')
    baseline_visualization(next(iter(train_images.values())),workdir/'private_figures')
    return dict(seed=SEED,n_experiments=len(experiments),candidates=chosen_dev,
                selected_pipeline=winner,validation_MAE=float(vr.iloc[0].validation_MAE),
                locked_test_predictions=len(locked),next_step='COMMIT_AND_PUSH_STAGE1')


def verify_lock_commit(workdir: Path, commit: str):
    """Aşama 1 kilit çıktılarının git commit'indeki BYTE içeriğiyle aynı olduğunu kanıtlar."""
    commit=commit.strip()
    if not re.fullmatch(r'[a-fA-F0-9]{7,40}',commit):
        raise ValueError('Tamamen gerçek git commit hash değeri girin.')
    files=('selected_frames.csv','experiments.csv','validation_results.csv','final_pipeline.json',
           'test_predictions_locked.csv','lock_manifest.json')
    for filename in files:
        local=(workdir/filename).read_bytes()
        show=subprocess.run(['git','show',f'{commit}:{filename}'],cwd=workdir,capture_output=True)
        if show.returncode:
            raise RuntimeError(f'Lock commit {filename} içermiyor; kilit commit doğru mu? {show.stderr.decode(errors="replace")[:200]}')
        if local != show.stdout:
            raise RuntimeError(f'{filename} dosyası kilit commit sonrasında değişmiş; test değerlendirmesi engellendi.')
    manifest=json.loads((workdir/'lock_manifest.json').read_text(encoding='utf-8'))
    if manifest['selected_frames_sha256'] != _filesha(workdir/'selected_frames.csv') or \
       manifest['pipeline_sha256'] != _filesha(workdir/'final_pipeline.json') or \
       manifest['test_predictions_sha256'] != _filesha(workdir/'test_predictions_locked.csv'):
        raise RuntimeError('SHA256 kilit manifesti uyuşmazlığı')
    return True


def generate_stage2_figures(ds: Dataset, test_df: pd.DataFrame, best, outdir: Path):
    outdir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(14,3.6))
    for block,ax in zip([1,2,3],axes):
        part=test_df[test_df.block==block].sort_values('frame')
        x=part.frame.to_numpy()
        ax.plot(x,part.reference_count,'o-',label='Referans',lw=2,ms=4)
        ax.plot(x,part.baseline_prediction,'s--',label='Baseline',lw=1.3,ms=4)
        ax.plot(x,part.final_prediction,'^-.',label='Nihai',lw=1.3,ms=4)
        ax.set_title(f'Test bloğu {block}',fontsize=10)
        ax.set_xlabel('Kare numarası');ax.set_ylabel('Hücre sayısı')
        ax.grid(alpha=.25);ax.set_xticks(x)
    axes[0].legend(fontsize=8)
    fig.tight_layout();_savefig(outdir/'sekil5_zamansal_uc_blok.png',fig)

    errs=np.abs(test_df.final_prediction-test_df.reference_count)
    best_idx=int(errs.idxmin());worst_idx=int(errs.drop(best_idx).idxmax())
    # Zorlayıcı örnek: en büyük hücre sayılı kare; çakışma olursa median hatalı kare kullan.
    tough_idx=int(test_df.reference_count.idxmax())
    if tough_idx in (best_idx,worst_idx):
        for i in errs.sort_values(ascending=False).index:
            if int(i) not in (best_idx,worst_idx):
                tough_idx=int(i);break
    frames=[(best_idx,'En düşük hata'),(worst_idx,'En yüksek hata'),(tough_idx,'Yüksek yoğunluk')]
    fig,axes=plt.subplots(2,3,figsize=(11.5,7.5))
    for j,(idx,label) in enumerate(frames):
        record=test_df.loc[idx]
        img=ds.image('02',int(record.frame));count,steps=process(img,best,True)
        _show_img(axes[0,j],to_u8(img,'stretch'),f'{label} | t{int(record.frame):03d}')
        _show_img(axes[1,j],steps['morphology'],f"Gerçek {int(record.reference_count)} / Tahmin {count}")
    fig.tight_layout();_savefig(outdir/'sekil6_ornek_basari_hata.png',fig)


def run_stage2(root: str | Path, workdir: str | Path, lock_commit: str):
    """Gerçek git lock commit doğrulandıktan SONRA test referansları açılır."""
    workdir=Path(workdir).resolve()
    verify_lock_commit(workdir,lock_commit)
    if (workdir/'test_results.csv').exists():
        raise RuntimeError('Test değerlendirmesi zaten yapıldı. Yeniden çalıştırarak sonuçları değiştirmeyin.')
    ds=discover_dataset(root)
    locked=pd.read_csv(workdir/'test_predictions_locked.csv')
    selected=pd.read_csv(workdir/'selected_frames.csv')
    expected=selected[selected.group=='test'].sort_values(['block','frame']).frame.astype(int).tolist()
    if locked.frame.astype(int).tolist()!=expected or len(locked)!=15:
        raise RuntimeError('Test frame sırası veya sayısı seçilen split ile uyuşmuyor')
    pipe=json.loads((workdir/'final_pipeline.json').read_text(encoding='utf-8'))
    if pipe['selection_sha256'] != _filesha(workdir/'selected_frames.csv'):
        raise RuntimeError('Nihai pipeline split hash uyuşmuyor')

    # KİLİTLENMİŞ TAHMİNLERDEN SONRA TEST GT İLK KEZ OKUNUR.
    reference=[ds.reference('02',int(x)) for x in locked.frame]
    result=locked.copy()
    result['reference_count']=reference
    result['baseline_absolute_error']=np.abs(result.baseline_prediction-result.reference_count)
    result['final_absolute_error']=np.abs(result.final_prediction-result.reference_count)
    result['baseline_signed_error']=result.baseline_prediction-result.reference_count
    result['final_signed_error']=result.final_prediction-result.reference_count
    result.to_csv(workdir/'test_results.csv',index=False)

    rows=[]
    for method,pred,timing in [('Baseline','baseline_prediction','baseline_seconds'),
                               ('Nihai','final_prediction','final_seconds')]:
        metric=scores(result.reference_count,result[pred])
        rows.append(dict(method=method,**metric,mean_seconds=float(result[timing].mean())))
    summary=pd.DataFrame(rows)
    summary.to_csv(workdir/'test_metrics.csv',index=False)
    generate_stage2_figures(ds,result,pipe['config'],workdir/'private_figures')
    return dict(commit=lock_commit,test_frames=len(result),final_MAE=float(summary.iloc[1]['MAE']),
                baseline_MAE=float(summary.iloc[0]['MAE']),next_step='GENERATE_REPORT_AND_COMMIT_STAGE2')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description='Klasik hücre sayımı ödevinin iki aşamalı çalıştırıcısı')
    p.add_argument('stage',choices=['1','2'])
    p.add_argument('--data',default='data/Fluo-N2DH-GOWT1')
    p.add_argument('--workdir',default='.')
    p.add_argument('--lock-commit',default='')
    opt=p.parse_args()
    if opt.stage=='1': print(json.dumps(run_stage1(opt.data,opt.workdir),ensure_ascii=False,indent=2))
    else: print(json.dumps(run_stage2(opt.data,opt.workdir,opt.lock_commit),ensure_ascii=False,indent=2))
