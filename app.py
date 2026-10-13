import json
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from core import validate_artifact,predict,drift,fingerprint

@asynccontextmanager
async def lifespan(app):
    app.state.model=validate_artifact(json.loads(Path(__file__).with_name('model.json').read_text()))
    yield
app=FastAPI(title='ModelOps Watch',version='1.0.0',lifespan=lifespan)
class Batch(BaseModel):
    rows:list[list[float]]=Field(min_length=1,max_length=1000)

@app.get('/health')
def health():return {'status':'ok','model_sha256':fingerprint(app.state.model)}

@app.post('/predict')
def inference(q:Batch):
    try:return predict(app.state.model,q.rows)
    except ValueError as e:raise HTTPException(422,str(e)) from e

@app.post('/drift')
def monitor(q:Batch):
    try:return drift(app.state.model,q.rows)
    except ValueError as e:raise HTTPException(422,str(e)) from e

@app.get('/',include_in_schema=False)
def demo_page():
    from fastapi.responses import FileResponse
    from pathlib import Path
    return FileResponse(Path(__file__).with_name('index.html'))
