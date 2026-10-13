"""Deterministic classification with JSON artifacts and distribution monitoring."""
import hashlib
import json
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss

def validate_artifact(a):
    required={'version','feature_names','mean','scale','weights','intercept','classes','reference_bins','reference_frequencies','metrics'}
    if not required.issubset(a) or a['version']!=1:raise ValueError('invalid artifact schema')
    n=len(a['feature_names'])
    if n!=13 or len(a['mean'])!=n or len(a['scale'])!=n:raise ValueError('invalid feature schema')
    w=np.asarray(a['weights'],float)
    if w.shape!=(3,n) or len(a['intercept'])!=3 or len(a['classes'])!=3:raise ValueError('invalid model dimensions')
    for field in ['mean','scale','weights','intercept']:
        if not np.isfinite(np.asarray(a[field],float)).all():raise ValueError('nonfinite model parameters')
    if any(s<=0 for s in a['scale']):raise ValueError('scales must be positive')
    if len(a['reference_bins'])!=n or len(a['reference_frequencies'])!=n:raise ValueError('invalid drift schema')
    for bins,freq in zip(a['reference_bins'],a['reference_frequencies']):
        if len(bins)!=11 or len(freq)!=10 or not np.isfinite(bins).all() or not np.isfinite(freq).all():raise ValueError('invalid drift dimensions')
        if np.any(np.diff(bins)<=0) or np.any(np.asarray(freq)<0) or not np.isclose(sum(freq),1):raise ValueError('invalid drift reference')
    return a

def train(seed=42):
    d=load_wine()
    x_train,x_test,y_train,y_test=train_test_split(d.data,d.target,test_size=.25,stratify=d.target,random_state=seed)
    scaler=StandardScaler().fit(x_train)
    model=LogisticRegression(max_iter=2000,random_state=seed).fit(scaler.transform(x_train),y_train)
    bins=[]; frequencies=[]
    for col in x_train.T:
        edges=np.quantile(col,np.linspace(0,1,11))
        # Wine features have repeated values; monotonic bins avoid ambiguous histograms.
        for i in range(1,len(edges)):
            edges[i]=max(edges[i],edges[i-1]+1e-8)
        bins.append(edges.tolist())
        frequencies.append((np.histogram(col,bins=edges)[0]/len(col)).tolist())
    p=model.predict_proba(scaler.transform(x_test))
    artifact={'version':1,'seed':seed,'dataset':'scikit-learn Wine (UCI), 178 samples; 133 train, 45 held-out test',
              'feature_names':d.feature_names,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
              'weights':model.coef_.tolist(),'intercept':model.intercept_.tolist(),'classes':model.classes_.tolist(),
              'reference_bins':bins,'reference_frequencies':frequencies,
              'metrics':{'test_accuracy':float(accuracy_score(y_test,p.argmax(axis=1))),
                         'test_log_loss':float(log_loss(y_test,p))}}
    validate_artifact(artifact)
    return artifact,x_test,model,scaler

def fingerprint(a):
    return hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def predict(a,rows):
    validate_artifact(a)
    x=np.asarray(rows,float)
    if x.ndim!=2 or x.shape[1]!=13 or not np.isfinite(x).all() or not 1<=len(x)<=1000:
        raise ValueError('need 1 to 1000 finite rows of 13 features')
    z=(x-np.asarray(a['mean']))/np.asarray(a['scale'])
    logits=z@np.asarray(a['weights']).T+np.asarray(a['intercept'])
    logits-=logits.max(axis=1,keepdims=True)
    p=np.exp(logits);p/=p.sum(axis=1,keepdims=True)
    return {'model_sha256':fingerprint(a),'classes':np.asarray(a['classes'])[p.argmax(axis=1)].tolist(),'probabilities':p.tolist()}

def drift(a,rows,threshold=.2):
    predict(a,rows) # validation shared with inference
    x=np.asarray(rows,float)
    if len(x)<20:raise ValueError('drift monitoring needs at least 20 rows')
    results=[]
    for name,col,edges,reference in zip(a['feature_names'],x.T,a['reference_bins'],a['reference_frequencies']):
        e=np.asarray(edges);e[0]=-np.inf;e[-1]=np.inf
        current=np.histogram(col,bins=e)[0]/len(col)
        ref=np.maximum(reference,1e-6);cur=np.maximum(current,1e-6)
        ref/=ref.sum();cur/=cur.sum()
        psi=float(np.sum((cur-ref)*np.log(cur/ref)))
        results.append({'feature':name,'psi':psi,'alert':psi>threshold})
    return {'samples':len(x),'threshold':threshold,'features':results,'action':'review distributions; drift alone does not justify retraining'}
