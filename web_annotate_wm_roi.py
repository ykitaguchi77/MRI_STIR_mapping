"""
web_annotate_wm_roi.py — smartphone-friendly browser version of annotate_wm_roi.py.

Same task and the SAME annotation-file format (so compare_wm_manual_vs_auto.py
works unchanged): one slice per case, one or more ellipses on white matter, mean
raw STIR signal recorded. Served with the Python standard library behind HTTP
Basic authentication; expose it with a Cloudflare quick tunnel:

    python web_annotate_wm_roi.py --dicom ../MRI_TOM/HANDAI_STIR_Coronal \
        --out wm_manual_roi_orbit/annotations.json --n 20 --orbit-offset 3 \
        --password <secret> --port 8765
    cloudflared tunnel --url http://localhost:8765

Touch controls: drag on empty image = new ellipse, drag on an ellipse = move it,
buttons resize / delete the selected ellipse, change slice / case, window, zoom.
"""

from __future__ import annotations

import argparse
import base64
import hmac
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

import sir_analysis as S
from annotate_wm_roi import list_series, pick_cases, roi_stats

PAGE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>WM ROI</title>
<style>
 body{margin:0;font-family:system-ui,sans-serif;background:#111;color:#eee}
 #top{padding:6px 8px;font-size:13px}
 #wrap{position:relative;overflow:hidden;width:100vw;max-width:640px;margin:auto;background:#000}
 canvas{display:block;touch-action:none;width:100%}
 .row{display:flex;gap:4px;padding:4px 6px;flex-wrap:wrap;max-width:640px;margin:auto}
 button{flex:1;min-width:52px;padding:11px 4px;font-size:15px;border:0;border-radius:6px;background:#333;color:#eee}
 button.on{background:#2a7}
 button.warn{background:#844}
 button.go{background:#257}
 #stat{padding:4px 8px;font-size:13px;color:#9f9;min-height:34px;max-width:640px;margin:auto}
 #help{padding:6px 8px;font-size:12px;color:#888;max-width:640px;margin:auto}
</style></head><body>
<div id="top">読み込み中…</div>
<div id="wrap"><canvas id="cv"></canvas></div>
<div id="stat"></div>
<div class="row">
 <button onclick="slice(-1)">スライス▲</button><button onclick="slice(1)">スライス▼</button>
 <button onclick="win(-1)">暗</button><button onclick="win(1)">明</button>
 <button id="zb" onclick="toggleZoom()">拡大</button><button id="pb" onclick="togglePan()">パン</button>
</div>
<div class="row">
 <button onclick="rs('rx',1/1.15)">幅−</button><button onclick="rs('rx',1.15)">幅＋</button>
 <button onclick="rs('ry',1/1.15)">高−</button><button onclick="rs('ry',1.15)">高＋</button>
 <button class="warn" onclick="del()">削除</button>
</div>
<div class="row">
 <button onclick="go(-1)">◀ 前の症例</button>
 <button class="go" onclick="save(true)">保存して次へ ▶</button>
</div>
<div class="row"><button onclick="save(false)">保存</button><button onclick="jumpTodo()">未完了へ</button></div>
<div id="help">画像の空いた所をドラッグ＝楕円を作成、楕円をドラッグ＝移動（選択）。選択中の楕円はボタンで幅・高さ変更／削除。
1症例につき1スライスのみ：楕円を置いた後にスライスを変えると楕円は消えます。外眼筋が見えるスライスの前頭葉白質に置いてください。</div>
<script>
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
let st=null,idx=0,cs=null,z=0,ells=[],sel=-1,img=new Image(),winp=99.5,zoom=1,pan=false,ox=0,oy=0,drag=null,dirty=false;
const SCALE=3;  // canvas pixels per image pixel
async function api(p,opt){const r=await fetch(p,opt);if(!r.ok)throw new Error(await r.text());return r.json();}
async function init(){st=await api('api/state');const t=st.cases.findIndex(c=>!st.done.includes(c));await load(t<0?0:t);}
async function load(i){idx=i;cs=await api('api/case?i='+i);z=cs.slice;ells=cs.ellipses.map(e=>({...e}));sel=-1;dirty=false;
 cv.width=cs.W*SCALE;cv.height=cs.H*SCALE;ox=0;oy=0;await loadImg();}
function loadImg(){return new Promise(res=>{img=new Image();img.onload=()=>{draw();res();};
 img.src='api/slice?i='+idx+'&z='+z+'&w='+winp;});}
function view(){return {s:SCALE*zoom,ox:ox,oy:oy};}
function toImg(px,py){const v=view();return [(px-v.ox)/v.s,(py-v.oy)/v.s];}
function draw(){const v=view();ctx.fillStyle='#000';ctx.fillRect(0,0,cv.width,cv.height);ctx.imageSmoothingEnabled=false;
 ctx.drawImage(img,v.ox,v.oy,cs.W*v.s,cs.H*v.s);
 ells.forEach((e,k)=>{ctx.beginPath();ctx.ellipse(v.ox+(e.cx+.5)*v.s,v.oy+(e.cy+.5)*v.s,e.rx*v.s,e.ry*v.s,0,0,2*Math.PI);
  ctx.lineWidth=k==sel?4:2.5;ctx.strokeStyle=k==sel?'#ff0':'#0f6';ctx.stroke();});
 if(drag&&drag.t=='new'){const [a,b]=[drag.x0,drag.y0],[c,d]=[drag.x1,drag.y1];ctx.strokeStyle='#ff0';ctx.lineWidth=2;
  ctx.beginPath();ctx.ellipse((a+c)/2,(b+d)/2,Math.abs(c-a)/2,Math.abs(d-b)/2,0,0,2*Math.PI);ctx.stroke();}
 const done=st.done.includes(cs.case);
 document.getElementById('top').textContent=`症例 ${idx+1}/${st.cases.length}  ${cs.case.replace('_STIR_Coronal','')}  [${done?'済':'未'}]  スライス ${z}/${cs.nz-1}  完了 ${st.done.length}/${st.cases.length}`+(dirty?'  *未保存':'');
 stats();}
let statT=null;function stats(){clearTimeout(statT);const el=document.getElementById('stat');
 if(!ells.length){el.textContent='楕円なし';return;}
 statT=setTimeout(async()=>{const r=await api('api/stats',{method:'POST',body:JSON.stringify({i:idx,z:z,ellipses:ells})});
  el.textContent=`ROI平均 ${r.mean.toFixed(1)} ± ${r.std.toFixed(1)}  n=${r.n}  (楕円${ells.length}個)`;},150);}
function pt(ev){const r=cv.getBoundingClientRect();const t=ev.touches?ev.touches[0]||ev.changedTouches[0]:ev;
 return [(t.clientX-r.left)*cv.width/r.width,(t.clientY-r.top)*cv.height/r.height];}
function hit(x,y){const [ix,iy]=toImg(x,y);for(let k=ells.length-1;k>=0;k--){const e=ells[k];
 const rx=Math.max(e.rx,3),ry=Math.max(e.ry,3);if(((ix-e.cx-.5)/rx)**2+((iy-e.cy-.5)/ry)**2<=1)return k;}return -1;}
function down(ev){ev.preventDefault();const [x,y]=pt(ev);
 if(pan){drag={t:'pan',x,y};return;}
 const k=hit(x,y);if(k>=0){sel=k;drag={t:'move',k,x,y};}else drag={t:'new',x0:x,y0:y,x1:x,y1:y};draw();}
function move(ev){if(!drag)return;ev.preventDefault();const [x,y]=pt(ev);const v=view();
 if(drag.t=='pan'){ox+=x-drag.x;oy+=y-drag.y;drag.x=x;drag.y=y;}
 else if(drag.t=='move'){const e=ells[drag.k];e.cx+=(x-drag.x)/v.s;e.cy+=(y-drag.y)/v.s;drag.x=x;drag.y=y;dirty=true;}
 else{drag.x1=x;drag.y1=y;}draw();}
function up(ev){if(!drag)return;ev.preventDefault();
 if(drag.t=='new'){const [ax,ay]=toImg(Math.min(drag.x0,drag.x1),Math.min(drag.y0,drag.y1));
  const [bx,by]=toImg(Math.max(drag.x0,drag.x1),Math.max(drag.y0,drag.y1));const rx=(bx-ax)/2,ry=(by-ay)/2;
  if(rx>=1.5&&ry>=1.5){ells.push({cx:(ax+bx)/2-.5,cy:(ay+by)/2-.5,rx,ry});sel=ells.length-1;dirty=true;}}
 drag=null;draw();}
cv.addEventListener('touchstart',down,{passive:false});cv.addEventListener('touchmove',move,{passive:false});
cv.addEventListener('touchend',up,{passive:false});cv.addEventListener('mousedown',down);
window.addEventListener('mousemove',move);window.addEventListener('mouseup',up);
function rs(k,f){if(sel<0)return;ells[sel][k]=Math.max(1.5,ells[sel][k]*f);dirty=true;draw();}
function del(){if(sel<0)return;ells.splice(sel,1);sel=-1;dirty=true;draw();}
async function slice(d){const nz=Math.min(cs.nz-1,Math.max(0,z+d));if(nz==z)return;
 if(ells.length&&!confirm('楕円を消してスライスを変えますか？'))return;
 if(ells.length){ells=[];sel=-1;dirty=true;}z=nz;await loadImg();}
async function win(d){winp=Math.min(100,Math.max(90,winp+d*0.5));await loadImg();}
function toggleZoom(){const c=cv.width/2,m=cv.height/2;const [ix,iy]=toImg(c,m);zoom=zoom==1?2:1;
 const s=SCALE*zoom;ox=c-ix*s;oy=m-iy*s;if(zoom==1){ox=0;oy=0;}
 document.getElementById('zb').classList.toggle('on',zoom==2);draw();}
function togglePan(){pan=!pan;document.getElementById('pb').classList.toggle('on',pan);}
async function save(next){const r=await api('api/save',{method:'POST',body:JSON.stringify({i:idx,z:z,ellipses:ells})});
 st.done=r.done;dirty=false;if(next){if(idx+1<st.cases.length)await load(idx+1);else{draw();alert('最後の症例です');}}else draw();}
async function go(d){if(dirty&&!confirm('未保存の変更を破棄しますか？'))return;const n=idx+d;if(n>=0&&n<st.cases.length)await load(n);}
async function jumpTodo(){const t=st.cases.findIndex(c=>!st.done.includes(c));if(t>=0)await load(t);}
init().catch(e=>{document.getElementById('top').textContent='エラー: '+e;});
</script></body></html>
"""


class Store:
    """Volumes (cached), annotation file and its lock."""

    def __init__(self, dicom: str, cases: list[str], out: Path, orbit_offset: int | None):
        self.dicom, self.cases, self.out, self.orbit_offset = dicom, cases, out, orbit_offset
        self.lock = threading.Lock()
        self.vols: dict[str, np.ndarray] = {}
        self.ann = {}
        if out.exists():
            self.ann = json.loads(out.read_text(encoding="utf-8")).get("annotations", {})

    def vol(self, case: str) -> np.ndarray:
        with self.lock:
            if case not in self.vols:
                self.vols[case], _ = S.load_dicom_series(self.dicom, case)
            return self.vols[case]

    def default_slice(self, v: np.ndarray) -> int:
        import orbital_sir_map as O
        try:
            if self.orbit_offset is not None:
                _, gz = O.detect_globes(v)
                d = -1 if gz > v.shape[2] / 2 else 1
                return int(np.clip(gz + d * self.orbit_offset, 0, v.shape[2] - 1))
            brain, _ = O.auto_orbit_and_brain_slices(v)
            return int(np.median(brain))
        except Exception:  # noqa: BLE001
            return v.shape[2] // 2

    def save(self, case: str, z: int, ellipses: list[dict], user: str):
        v = self.vol(case)
        with self.lock:
            if ellipses:
                self.ann[case] = {
                    "slice": int(z),
                    "ellipses": [{k: round(float(e[k]), 2) for k in ("cx", "cy", "rx", "ry")}
                                 for e in ellipses],
                    "stats": roi_stats(v[:, :, z], ellipses),
                    "annotator": user,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            else:
                self.ann.pop(case, None)
            self.out.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.out.with_suffix(".tmp")
            tmp.write_text(json.dumps({"dicom": str(self.dicom), "cases": self.cases,
                                       "annotations": self.ann}, indent=1, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(self.out)


def make_handler(store: Store, user: str, password: str):
    token = base64.b64encode(f"{user}:{password}".encode()).decode()

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):  # quiet; keep only errors
            pass

        def _auth(self) -> bool:
            got = self.headers.get("Authorization", "")
            if got.startswith("Basic ") and hmac.compare_digest(got[6:].strip(), token):
                return True
            time.sleep(1.0)  # slow down guessing
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="WM ROI", charset="UTF-8"')
            self.end_headers()
            return False

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(json.dumps(obj).encode(), "application/json", code)

        def _case(self, q) -> str:
            return store.cases[int(q["i"][0])]

        def do_GET(self):
            if not self._auth():
                return
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path in ("/", "/index.html"):
                    self._send(PAGE.encode(), "text/html; charset=utf-8")
                elif u.path == "/api/state":
                    self._json({"cases": store.cases, "done": list(store.ann)})
                elif u.path == "/api/case":
                    case = self._case(q)
                    v = store.vol(case)
                    rec = store.ann.get(case)
                    self._json({"case": case, "nz": v.shape[2], "H": v.shape[0], "W": v.shape[1],
                                "slice": rec["slice"] if rec else store.default_slice(v),
                                "ellipses": rec["ellipses"] if rec else []})
                elif u.path == "/api/slice":
                    v = store.vol(self._case(q))
                    z = int(np.clip(int(q["z"][0]), 0, v.shape[2] - 1))
                    w = float(q.get("w", ["99.5"])[0])
                    vmax = float(np.percentile(v, np.clip(w, 90, 100)))
                    img8 = (np.clip(v[:, :, z], 0, vmax) / max(vmax, 1e-6) * 255).astype(np.uint8)
                    buf = io.BytesIO()
                    Image.fromarray(img8).save(buf, format="PNG")
                    self._send(buf.getvalue(), "image/png")
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as exc:  # noqa: BLE001
                self._json({"error": str(exc)}, 400)

        def do_POST(self):
            if not self._auth():
                return
            u = urlparse(self.path)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                case = store.cases[int(body["i"])]
                v = store.vol(case)
                z = int(np.clip(int(body["z"]), 0, v.shape[2] - 1))
                ells = [{k: float(e[k]) for k in ("cx", "cy", "rx", "ry")} for e in body["ellipses"]]
                if u.path == "/api/stats":
                    st = roi_stats(v[:, :, z], ells)
                    self._json({k: st[k] for k in ("mean", "std", "n")})
                elif u.path == "/api/save":
                    store.save(case, z, ells, user)
                    self._json({"ok": True, "done": list(store.ann)})
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as exc:  # noqa: BLE001
                self._json({"error": str(exc)}, 400)

    return H


def main():
    p = argparse.ArgumentParser(description="Browser (smartphone) WM ROI annotation")
    p.add_argument("--dicom", required=True)
    p.add_argument("--out", default="wm_manual_roi_orbit/annotations.json")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--orbit-offset", type=int, help="start N slices behind the eyeball slice")
    p.add_argument("--user", default="reader")
    p.add_argument("--password", required=True)
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    out = Path(args.out)
    if out.exists():
        cases = json.loads(out.read_text(encoding="utf-8"))["cases"]
    else:
        cases = pick_cases(list_series(args.dicom), args.n, args.seed)
    store = Store(args.dicom, cases, out, args.orbit_offset)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store, args.user, args.password))
    print(f"Serving {len(cases)} cases on http://127.0.0.1:{args.port}  (user '{args.user}')", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
