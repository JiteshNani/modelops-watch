import copy
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app import app
from core import train,predict,drift,validate_artifact,fingerprint

@pytest.fixture(scope='module')
def trained():return train()

def test_json_inference_matches_sklearn(trained):
    a,x,m,s=trained
    np.testing.assert_allclose(predict(a,x)['probabilities'],m.predict_proba(s.transform(x)),atol=1e-10)

def test_nan_rejected(trained):
    with pytest.raises(ValueError):predict(trained[0],[[np.nan]*13])

def test_wrong_features_rejected(trained):
    with pytest.raises(ValueError):predict(trained[0],[[1,2]])

def test_shift_detected(trained):
    a,x,_,_=trained
    shifted=x.copy();shifted[:,0]+=4
    assert drift(a,shifted)['features'][0]['alert']

def test_small_batch_not_used_for_drift(trained):
    with pytest.raises(ValueError):drift(trained[0],trained[1][:2])

def test_tampered_scale_rejected(trained):
    a=copy.deepcopy(trained[0]);a['scale'][0]=0
    with pytest.raises(ValueError):validate_artifact(a)

def test_fingerprint_changes(trained):
    a=copy.deepcopy(trained[0]);before=fingerprint(a);a['intercept'][0]+=.1
    assert fingerprint(a)!=before

def test_api(trained):
    with TestClient(app) as c:
        assert c.post('/predict',json={'rows':trained[1][:1].tolist()}).status_code==200
        assert c.post('/predict',json={'rows':[[1]]}).status_code==422
