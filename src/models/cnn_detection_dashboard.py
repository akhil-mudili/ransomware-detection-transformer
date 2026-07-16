"""
cnn_detection_dashboard.py - CNN Baseline (Light Theme)
Author: Akhil Mudili | University of Galway
Run: python cnn_detection_dashboard.py -> opens http://localhost:5001
"""
import os,json,pickle,threading,webbrowser,time,queue,numpy as np,torch,torch.nn as nn
from flask import Flask,Response,render_template_string

BASE_DIR=r"D:\ACS\Final Project\ransomware-detection-transformer"
VOCAB_PATH=os.path.join(BASE_DIR,"data","vocabulary.pkl")
MODELS_DIR=os.path.join(BASE_DIR,"results","models")
RESULTS_DIR=os.path.join(BASE_DIR,"results","live_detection")
os.makedirs(RESULTS_DIR,exist_ok=True)

MAX_SEQ_LEN=3000;WINDOW_SIZES=[0.25,0.50,0.75,1.0];DEVICE=torch.device("cpu")
EMBED_DIM=64;NUM_FILTERS=128;KERNEL_SIZE=5;FC_DIM=128;DROPOUT=0.3

with open(VOCAB_PATH,"rb") as f: vocab_data=pickle.load(f)
api_to_id=vocab_data["api_to_id"];PAD_ID=vocab_data["pad_id"]
UNK_ID=api_to_id.get("<UNK>",1);VOCAB_SIZE=vocab_data["vocab_size"]

class CNNBaseline(nn.Module):
    def __init__(self,vocab_size,embed_dim,num_filters,kernel_size,fc_dim,dropout,max_seq_len):
        super(CNNBaseline,self).__init__()
        self.embedding=nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_dim,padding_idx=0)
        self.conv1=nn.Conv1d(in_channels=embed_dim,out_channels=num_filters,kernel_size=kernel_size,padding=kernel_size//2)
        self.pool1=nn.MaxPool1d(kernel_size=2)
        self.conv2=nn.Conv1d(in_channels=num_filters,out_channels=num_filters*2,kernel_size=kernel_size,padding=kernel_size//2)
        self.pool2=nn.MaxPool1d(kernel_size=2)
        conv_out_len=max_seq_len//4
        self.flat_size=(num_filters*2)*conv_out_len
        self.fc1=nn.Linear(self.flat_size,fc_dim);self.dropout=nn.Dropout(dropout)
        self.fc2=nn.Linear(fc_dim,2);self.relu=nn.ReLU()
    def forward(self,x):
        x=self.embedding(x);x=x.permute(0,2,1)
        x=self.relu(self.conv1(x));x=self.pool1(x)
        x=self.relu(self.conv2(x));x=self.pool2(x)
        x=x.view(x.size(0),-1);x=self.relu(self.fc1(x))
        x=self.dropout(x);x=self.fc2(x);return x

def load_models():
    models={}
    for ws in WINDOW_SIZES:
        label=int(ws*100)
        path=os.path.join(MODELS_DIR,f"cnn_window_{label}.pt")
        model=CNNBaseline(vocab_size=VOCAB_SIZE,embed_dim=EMBED_DIM,num_filters=NUM_FILTERS,kernel_size=KERNEL_SIZE,fc_dim=FC_DIM,dropout=DROPOUT,max_seq_len=MAX_SEQ_LEN).to(DEVICE)
        model.load_state_dict(torch.load(path,map_location=DEVICE))
        model.eval();models[ws]=model
    return models

def extract_api_calls(json_path):
    with open(json_path,"r",encoding="utf-8",errors="ignore") as f: report=json.load(f)
    calls=[]
    for p in report.get("behavior",{}).get("processes",[]):
        for c in p.get("calls",[]):
            if c.get("api"): calls.append(c["api"])
    return calls

def encode_and_pad(calls):
    enc=[api_to_id.get(c,UNK_ID) for c in calls]
    if len(enc)>MAX_SEQ_LEN: enc=enc[:MAX_SEQ_LEN]
    else: enc=enc+[PAD_ID]*(MAX_SEQ_LEN-len(enc))
    return enc

event_queue=queue.Queue()
def send_event(data): event_queue.put(json.dumps(data))

def run_detection(models):
    VT_BASE=r"D:\ACS\Final Project\ransomware dataset\vt_cuckoo_reports_final"
    configs=[
        {"folder":r"D:\ACS\Final Project\ransomware dataset\some ransomware cuckoo analysis report\darkside","family":"darkside","zeroday":False},
        {"folder":r"D:\ACS\Final Project\ransomware dataset\some ransomware cuckoo analysis report\locky\locky","family":"locky","zeroday":False},
        {"folder":r"D:\ACS\Final Project\ransomware dataset\some ransomware cuckoo analysis report\ryuk","family":"ryuk","zeroday":False},
        {"folder":r"D:\ACS\Final Project\ransomware dataset\some ransomware cuckoo analysis report\reveton","family":"reveton","zeroday":False},
        {"folder":r"D:\ACS\Final Project\ransomware dataset\some ransomware cuckoo analysis report\wannacry\wannacry","family":"wannacry","zeroday":True},
        {"folder":r"D:\ACS\Final Project\ransomware dataset\some ransomware cuckoo analysis report\sodinokibi\sodinokibi","family":"sodinokibi","zeroday":True},
        {"folder":os.path.join(VT_BASE,"crowti"),"family":"crowti","zeroday":True},
        {"folder":os.path.join(VT_BASE,"cryptodef"),"family":"cryptodef","zeroday":True},
        {"folder":os.path.join(VT_BASE,"ctblocker"),"family":"ctblocker","zeroday":True},
    ]
    all_results=[];send_event({"type":"start"});time.sleep(0.5)
    for cfg in configs:
        if not os.path.exists(cfg["folder"]): continue
        files=[f for f in os.listdir(cfg["folder"]) if f.endswith(".json")][:3]
        if not files: continue
        send_event({"type":"family_start","family":cfg["family"],"zeroday":cfg["zeroday"]});time.sleep(0.3)
        for fname in files:
            calls=extract_api_calls(os.path.join(cfg["folder"],fname))
            total=len(calls)
            if total<10: continue
            send_event({"type":"sample_start","family":cfg["family"],"zeroday":cfg["zeroday"],"filename":fname[:50],"total_calls":total});time.sleep(0.4)
            window_results=[]
            for ws in WINDOW_SIZES:
                label=int(ws*100);n_calls=max(1,int(total*ws))
                enc=encode_and_pad(calls[:n_calls])
                tensor=torch.tensor([enc],dtype=torch.long).to(DEVICE)
                with torch.no_grad():
                    output=models[ws](tensor);probs=torch.softmax(output,dim=1)
                    prediction=output.argmax(dim=1).item();confidence=probs[0][prediction].item()
                verdict="RANSOMWARE" if prediction==1 else "BENIGN"
                correct=prediction==1
                send_event({"type":"window_result","family":cfg["family"],"filename":fname[:50],"window":label,"n_calls":n_calls,"verdict":verdict,"confidence":round(confidence*100,1),"correct":correct})
                time.sleep(0.6);window_results.append({"window":label,"verdict":verdict,"confidence":round(confidence*100,1),"correct":correct})
            all_results.append({"family":cfg["family"],"zeroday":cfg["zeroday"],"filename":fname,"total_calls":total,"windows":window_results})
    send_event({"type":"complete","results":all_results})
    with open(os.path.join(RESULTS_DIR,"cnn_dashboard_results.json"),"w") as f: json.dump(all_results,f,indent=2)

app=Flask(__name__)

HTML=r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CNN Ransomware Detection</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f4f4f4;color:#1a1a1a;min-height:100vh}
.header{background:#fff;border-bottom:1px solid #e0e0e0;padding:20px 40px;display:flex;align-items:center;gap:16px;box-shadow:0 1px 4px rgba(0,0,0,.07)}
.header-icon{width:44px;height:44px;background:#fff3e0;border:1.5px solid #e65100;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:22px}
.header h1{font-size:20px;font-weight:600;color:#111}
.header p{font-size:13px;color:#888;margin-top:2px}
.model-badge{padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;background:#fff3e0;color:#e65100;border:1px solid #ffcc80;margin-left:8px}
.status-pill{margin-left:auto;padding:6px 16px;border-radius:20px;font-size:12px;font-weight:500}
.status-pill.running{border:1px solid #f59e0b;color:#b45309;background:#fffbeb}
.status-pill.done{border:1px solid #16a34a;color:#15803d;background:#f0fdf4}
.main{padding:24px 40px;max-width:1200px;margin:0 auto}
.summary-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}
.metric{background:#fff;border:1px solid #e8e8e8;border-radius:12px;padding:18px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.metric-label{font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.6px;margin-bottom:10px}
.metric-val{font-size:30px;font-weight:700;color:#111}
.metric-sub{font-size:11px;color:#bbb;margin-top:4px}
.window-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}
.window-card{background:#fff;border:1px solid #e8e8e8;border-radius:12px;padding:16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.window-pct{font-size:12px;color:#aaa;margin-bottom:8px;font-weight:500}
.window-acc{font-size:24px;font-weight:700;color:#ccc}
.window-acc.green{color:#16a34a}.window-acc.yellow{color:#d97706}.window-acc.red{color:#dc2626}
.samples-title{font-size:14px;font-weight:600;color:#555;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.pulse{width:8px;height:8px;border-radius:50%;background:#d97706;animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.sample-card{background:#fff;border:1px solid #e8e8e8;border-radius:12px;margin-bottom:12px;overflow:hidden;animation:slideIn .3s ease;box-shadow:0 1px 3px rgba(0,0,0,.04)}
@keyframes slideIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.sample-head{display:flex;align-items:center;gap:10px;padding:12px 16px;border-bottom:1px solid #f2f2f2;background:#fafafa}
.family-tag{font-size:11px;font-weight:600;padding:3px 10px;border-radius:20px;text-transform:uppercase;letter-spacing:.3px}
.tag-darkside{background:#ede9fe;color:#5b21b6;border:1px solid #c4b5fd}
.tag-locky{background:#ede9fe;color:#5b21b6;border:1px solid #c4b5fd}
.tag-ryuk{background:#fee2e2;color:#b91c1c;border:1px solid #fca5a5}
.tag-reveton{background:#fef3c7;color:#92400e;border:1px solid #fcd34d}
.tag-wannacry{background:#fee2e2;color:#991b1b;border:1px solid #f87171}
.tag-sodinokibi{background:#fdf2f8;color:#86198f;border:1px solid #f0abfc}
.tag-crowti{background:#f0fdf4;color:#15803d;border:1px solid #86efac}
.tag-cryptodef{background:#eff6ff;color:#1d4ed8;border:1px solid #93c5fd}
.tag-ctblocker{background:#fff7ed;color:#c2410c;border:1px solid #fdba74}
.zd-tag{font-size:10px;padding:2px 8px;border-radius:20px;background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;margin-left:2px;font-weight:500}
.fname{font-size:12px;color:#bbb;font-family:monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:280px}
.calls-count{margin-left:auto;font-size:12px;color:#ccc}
.windows-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:#f0f0f0}
.win-cell{background:#fff;padding:12px 14px}
.win-pct{font-size:10px;color:#bbb;margin-bottom:6px;text-transform:uppercase;letter-spacing:.6px;font-weight:500}
.win-verdict{font-size:13px;font-weight:600;margin-bottom:7px;display:flex;align-items:center;gap:5px}
.win-verdict.ransomware{color:#dc2626}.win-verdict.benign{color:#16a34a}.win-verdict.wrong{color:#d97706}.win-verdict.pending{color:#ddd}
.bar-track{height:4px;background:#f0f0f0;border-radius:2px;overflow:hidden;margin-bottom:5px}
.bar-fill{height:100%;border-radius:2px;transition:width .8s ease}
.bar-fill.ransomware{background:#dc2626}.bar-fill.benign{background:#16a34a}.bar-fill.pending{width:0%;background:#eee}
.conf-text{font-size:11px;color:#ccc}
</style></head><body>
<div class="header">
  <div class="header-icon">🔍</div>
  <div>
    <h1>Ransomware Live Detection <span class="model-badge">CNN Baseline</span></h1>
    <p>Conv1D model · Behavioral API call sequence analysis · University of Galway</p>
  </div>
  <div class="status-pill running" id="status-pill">⚡ Running</div>
</div>
<div class="main">
  <div class="summary-row">
    <div class="metric"><div class="metric-label">Samples processed</div><div class="metric-val" id="m-processed">0</div><div class="metric-sub">live count</div></div>
    <div class="metric"><div class="metric-label">Detected correctly</div><div class="metric-val" id="m-correct" style="color:#16a34a">0</div><div class="metric-sub" id="m-correct-sub">all windows correct</div></div>
    <div class="metric"><div class="metric-label">Zero-day tested</div><div class="metric-val" id="m-zeroday" style="color:#1d4ed8">0</div><div class="metric-sub">5 unseen families</div></div>
    <div class="metric"><div class="metric-label">Current family</div><div class="metric-val" id="m-family" style="font-size:18px;color:#d97706">—</div><div class="metric-sub" id="m-zd-label"></div></div>
  </div>
  <div class="window-row">
    <div class="window-card"><div class="window-pct">25% window</div><div class="window-acc" id="w25">—</div></div>
    <div class="window-card"><div class="window-pct">50% window</div><div class="window-acc" id="w50">—</div></div>
    <div class="window-card"><div class="window-pct">75% window</div><div class="window-acc" id="w75">—</div></div>
    <div class="window-card"><div class="window-pct">100% window</div><div class="window-acc" id="w100">—</div></div>
  </div>
  <div class="samples-title"><span class="pulse" id="pulse-dot"></span> Live results</div>
  <div id="samples-container"></div>
  <div style="margin-top:20px;padding:14px 18px;background:#fff;border:1px solid #e8e8e8;border-radius:12px;display:flex;gap:24px;align-items:center;font-size:12px;color:#666;">
    <span style="font-weight:600;color:#444">Legend:</span>
    <span>🔴 Ransomware detected (correct)</span>
    <span>🟢 Benign detected (correct)</span>
    <span>⚠️ Wrong prediction</span>
    <span style="background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:500">ZERO-DAY</span> = family not seen during training
  </div>
</div>
<script>
const state={processed:0,windowCorrect:{25:0,50:0,75:0,100:0},windowTotal:{25:0,50:0,75:0,100:0},zdCount:0};
const es=new EventSource('/stream');
es.onmessage=e=>{
  const d=JSON.parse(e.data);
  if(d.type==='family_start'){
    document.getElementById('m-family').textContent=d.family;
    document.getElementById('m-zd-label').textContent=d.zeroday?'⚡ Zero-day family':'Known family';
    if(d.zeroday){state.zdCount++;document.getElementById('m-zeroday').textContent=state.zdCount;}
  }
  if(d.type==='sample_start'){
    state.processed++;
    document.getElementById('m-processed').textContent=state.processed;
    const safeId=d.filename.replace(/[^a-zA-Z0-9]/g,'-');
    const card=document.createElement('div');
    card.className='sample-card';card.id='card-'+safeId;
    card.innerHTML=`<div class="sample-head">
      <span class="family-tag tag-${d.family}">${d.family}</span>
      ${d.zeroday?'<span class="zd-tag">ZERO-DAY</span>':''}
      <span class="fname">${d.filename}</span>
      <span class="calls-count">${d.total_calls.toLocaleString()} API calls</span>
    </div><div class="windows-grid">${[25,50,75,100].map(p=>`
      <div class="win-cell" id="wp${p}-${safeId}">
        <div class="win-pct">${p}% window</div>
        <div class="win-verdict pending">Analysing...</div>
        <div class="bar-track"><div class="bar-fill pending"></div></div>
        <div class="conf-text">—</div>
      </div>`).join('')}</div>`;
    document.getElementById('samples-container').insertBefore(card,document.getElementById('samples-container').firstChild);
  }
  if(d.type==='window_result'){
    const safeId=d.filename.replace(/[^a-zA-Z0-9]/g,'-');
    const cell=document.getElementById('wp'+d.window+'-'+safeId);
    if(cell){
      const isRW=d.verdict==='RANSOMWARE',ok=d.correct;
      const vc=ok?(isRW?'ransomware':'benign'):'wrong';
      const icon=ok?(isRW?'🔴':'🟢'):'⚠️';
      cell.querySelector('.win-verdict').className='win-verdict '+vc;
      cell.querySelector('.win-verdict').innerHTML=icon+' '+(isRW?'Ransomware':'Benign')+(ok?'':' ⚠');
      const bar=cell.querySelector('.bar-fill');
      bar.className='bar-fill '+(isRW?'ransomware':'benign');
      bar.style.width=d.confidence+'%';
      cell.querySelector('.conf-text').textContent=d.confidence.toFixed(1)+'% confidence';
    }
    state.windowTotal[d.window]++;
    if(d.correct)state.windowCorrect[d.window]++;
    [25,50,75,100].forEach(p=>{
      const el=document.getElementById('w'+p);
      if(state.windowTotal[p]>0){
        const acc=Math.round(state.windowCorrect[p]/state.windowTotal[p]*100);
        el.textContent=state.windowCorrect[p]+'/'+state.windowTotal[p]+' ('+acc+'%)';
        el.className='window-acc '+(acc===100?'green':acc>=80?'yellow':'red');
      }
    });
  }
  if(d.type==='complete'){
    document.getElementById('status-pill').className='status-pill done';
    document.getElementById('status-pill').textContent='✓ Complete';
    document.getElementById('pulse-dot').style.display='none';
    const allCorrect=d.results.filter(s=>s.windows.every(w=>w.correct)).length;
    const knownCorrect=d.results.filter(s=>!s.zeroday&&s.windows.every(w=>w.correct)).length;
    const knownTotal=d.results.filter(s=>!s.zeroday).length;
    const zdCorrect=d.results.filter(s=>s.zeroday&&s.windows.every(w=>w.correct)).length;
    const zdTotal=d.results.filter(s=>s.zeroday).length;
    document.getElementById('m-correct').textContent=allCorrect+'/'+d.results.length;
    document.getElementById('m-correct-sub').textContent='Known: '+knownCorrect+'/'+knownTotal+' | Zero-day: '+zdCorrect+'/'+zdTotal;
    document.getElementById('m-zeroday').textContent=zdTotal;
    es.close();
  }
};
</script></body></html>'''

@app.route('/')
def index(): return render_template_string(HTML)

@app.route('/stream')
def stream():
    def generate():
        while True:
            try:
                data=event_queue.get(timeout=30)
                yield f"data: {data}\n\n"
                if '"type": "complete"' in data: break
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'
    return Response(generate(),mimetype='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

if __name__=="__main__":
    print("Loading CNN models...")
    models=load_models()
    print("Models loaded. Starting CNN dashboard...")
    t=threading.Thread(target=lambda:(time.sleep(2),run_detection(models)),daemon=True)
    t.start()
    print("\n"+"="*55)
    print("  CNN Dashboard: http://localhost:5001")
    print("  Press Ctrl+C to stop")
    print("="*55+"\n")
    threading.Timer(1.5,lambda:webbrowser.open("http://localhost:5001")).start()
    app.run(host="0.0.0.0",port=5001,debug=False,threaded=True)
