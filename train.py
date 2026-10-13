import json
from pathlib import Path
from core import train,fingerprint,predict,drift

if __name__=='__main__':
    artifact,test,_,_=train()
    Path('model.json').write_text(json.dumps(artifact,indent=2))
    shifted=test.copy();shifted[:,0]+=4
    report={'metrics':artifact['metrics'],'model_sha256':fingerprint(artifact),
            'shifted_demo':drift(artifact,shifted),'note':'Small educational dataset; drift threshold is illustrative, not operationally calibrated.'}
    Path('evaluation.json').write_text(json.dumps(report,indent=2))
    Path('sample.json').write_text(json.dumps({'rows':test[:2].tolist()},indent=2))
    print(json.dumps(report['metrics'],indent=2))
