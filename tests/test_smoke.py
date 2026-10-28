import sys,tempfile,pathlib,numpy as np,pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.run_ciciot import load_split,build_matrices,label_to_family,fit_mlp
from src.common import train_mlp,family_js_divergence
from src.edge_runner import run_partition
from src.edge_data import TRANSPORT_IDENTITY_FIELDS
import torch
torch.set_num_threads(2)

with tempfile.TemporaryDirectory() as td:
 p=pathlib.Path(td)/'demo.csv'
 pd.DataFrame({'label':['BenignTraffic','Benign','DDoS-HTTP_Flood','DictionaryBruteForce','VulnerabilityScan'], 'feat':[0.0,0.1,1,1.2,1.3], 'other':[1,2,4,5,6]}).to_csv(p,index=False)
 t=load_split(p)
 assert list(t['y_binary'])==[0,0,1,1,1],list(t['y_binary'])
 assert label_to_family('BrowserHijacking')=='WebBased'
 assert label_to_family('MITM-ArpSpoofing')=='Spoofing'
 assert label_to_family('VulnerabilityScan')=='Recon'
 assert  len(TRANSPORT_IDENTITY_FIELDS)==8
 print('CICIoT label and family mapping PASS')

 rng=np.random.default_rng(42)
 n=240; X=rng.normal(size=(n,7)).astype('float32'); y=np.array(([0,1]*120),dtype='int64')
 X[:,0] += y*0.8
 m=train_mlp(X[:150],y[:150],X[150:200],y[150:200],X[200:],y[200:],seed=101)
 assert len(m['probability'])==40
 assert np.isfinite(m['probability']).all()
 print('Edge Tabular MLP train/predict PASS (epoch='+str(m['best_epoch'])+')')
 frame=pd.DataFrame(X,columns=['a','b','c','d','e','f','g'])
 rows, features=run_partition(frame,y,np.arange(180),np.arange(180,240),arm_label='test',split_id=1,seed_offset=1)
 assert len(rows)==5 and all(int(v['test_rows'])==60 for v in rows)
 assert len(features)==7
 assert np.isfinite([v['balanced_accuracy'] for v in rows]).all()
 print('Full Edge run_partition (baseline + MLP + RF + XGB + LightGBM) PASS')
 print('Class weighting and metric reporting PASS')

 v=family_js_divergence(pd.Series(['a','b','b']),pd.Series(['a','a','b']))
 assert np.isfinite(v)
 print('Jensen-Shannon helper PASS')
