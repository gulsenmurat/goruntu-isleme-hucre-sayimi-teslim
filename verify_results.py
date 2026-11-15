"""Çıktıları veri setini ve Jupyter'ı indirmeden doğrular; test parametrelerini değiştirmez."""
from pathlib import Path
import json
import hashlib
import math
import pandas as pd

root = Path(__file__).resolve().parent
sel = pd.read_csv(root/'selected_frames.csv')
assert len(sel) == 30 and sel['seed'].eq(11526).all()
assert sel['group'].value_counts().to_dict() == {'test':15,'development':10,'validation':5}
assert sel[['sequence','frame']].duplicated().sum() == 0
exp = pd.read_csv(root/'experiments.csv')
val = pd.read_csv(root/'validation_results.csv')
locked = pd.read_csv(root/'test_predictions_locked.csv')
test = pd.read_csv(root/'test_results.csv')
metrics = pd.read_csv(root/'test_metrics.csv').set_index('method')
assert len(exp)==24 and len(val)==3 and len(test)==15
assert not any('reference' in col for col in locked.columns), 'Kilit dosyasında referans olmamalı'
assert test['baseline_prediction'].equals(locked['baseline_prediction'])
assert test['final_prediction'].equals(locked['final_prediction'])
assert val['selected'].sum()==1 and val.loc[val['selected'],'pipeline'].iloc[0]=='E15_open3'
config=json.loads((root/'final_pipeline.json').read_text(encoding='utf-8'))
assert config['chosen_experiment']=='E15_open3'
manifest=json.loads((root/'lock_manifest.json').read_text(encoding='utf-8'))
for key, file in [('selected_frames_sha256','selected_frames.csv'),('pipeline_sha256','final_pipeline.json'),('test_predictions_sha256','test_predictions_locked.csv')]:
    assert manifest[key]==hashlib.sha256((root/file).read_bytes()).hexdigest(),file
for name, pred in [('Baseline','baseline_prediction'),('Nihai','final_prediction')]:
    errors=test[pred]-test['reference_count']
    expected={'MAE':errors.abs().mean(),'RMSE':math.sqrt((errors**2).mean()),
              'Bias':errors.mean(),'Exact-count Accuracy':errors.eq(0).mean()}
    for metric,value in expected.items():
        assert abs(metrics.loc[name,metric]-value)<1e-9,(name,metric)
    print(f"{name}: MAE={expected['MAE']:.4f}, RMSE={expected['RMSE']:.4f}, Bias={expected['Bias']:+.4f}")
print('DOĞRULAMA BAŞARILI — 24 deney, 3 aday, 15 kilitli test tahmini, CSV metrikleri ve SHA256 uyumlu.')
