# app.py
import base64, io, math
from typing import List, Tuple, Dict
import numpy as np
import cv2
from flask import Flask, request, jsonify

app = Flask(__name__)

def _b64_to_cv(img_b64: str):
    raw = base64.b64decode(img_b64.split(",")[-1], validate=False)
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

class DSU:
    def __init__(self, n): self.p=list(range(n)); self.r=[0]*n
    def find(self,x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self,a,b):
        pa,pb=self.find(a),self.find(b)
        if pa==pb: return False
        if self.r[pa]<self.r[pb]: pa,pb=pb,pa
        self.p[pb]=pa
        if self.r[pa]==self.r[pb]: self.r[pa]+=1
        return True

def _mst_weight(n_nodes: int, edges: List[Tuple[int,int,int]]) -> int:
    edges = sorted(edges, key=lambda e: e[2])
    dsu = DSU(n_nodes)
    total = used = 0
    for u,v,w in edges:
        if dsu.union(u,v):
            total += int(w); used += 1
            if used == n_nodes - 1: break
    return total

def _detect_nodes(img_bgr: np.ndarray) -> List[Tuple[int,int,int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9,9), 1.5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=25,
                               param1=100, param2=18, minRadius=6, maxRadius=28)
    out=[]
    if circles is not None:
        for c in np.uint16(np.around(circles))[0, :]:
            out.append((int(c[0]), int(c[1]), int(c[2])))
    if len(out)<3:
        _, bw = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)
        bw = cv2.medianBlur(bw, 5)
        cnts,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            (x,y), r = cv2.minEnclosingCircle(c)
            if 6<=r<=28: out.append((int(x),int(y),int(r)))
    dedup=[]
    for x,y,r in out:
        if all(_dist((x,y),(a,b))>10 for a,b,_ in dedup): dedup.append((x,y,r))
    return dedup

def _detect_line_segments(img_bgr: np.ndarray):
    g=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
    g=cv2.bilateralFilter(g,7,50,50)
    e=cv2.Canny(g,50,120,L2gradient=True)
    segs=cv2.HoughLinesP(e,1,np.pi/180,threshold=60,minLineLength=35,maxLineGap=8)
    return [] if segs is None else [tuple(map(int,s[0])) for s in segs]

def _segments_to_edges(nodes, segs, shape):
    centers=[(x,y) for x,y,_ in nodes]
    pairs=[]
    for i in range(len(centers)):
        for j in range(i+1,len(centers)):
            a,b=centers[i],centers[j]
            if _dist(a,b)<25: continue
            support=0
            for x1,y1,x2,y2 in segs:
                mx,my=(x1+x2)/2.0,(y1+y2)/2.0
                ax,ay=a; bx,by=b
                AB=np.array([bx-ax,by-ay],np.float32)
                AP=np.array([mx-ax,my-ay],np.float32)
                ABn=AB/(np.linalg.norm(AB)+1e-6)
                perp=np.linalg.norm(AP-ABn*np.dot(AP,ABn))
                on_span=0<=np.dot(AP,ABn)<=np.linalg.norm(AB)+5
                if on_span and perp<6.0: support+=1
            if support>=1:
                pairs.append((i,j,((centers[i][0]+centers[j][0])//2,
                                   (centers[i][1]+centers[j][1])//2)))
    return pairs

_DIGIT_TEMPLATES=None
def _make_digit_templates():
    global _DIGIT_TEMPLATES
    if _DIGIT_TEMPLATES is not None: return _DIGIT_TEMPLATES
    scales=[0.8,1.0,1.2]; thicks=[2,3]; templates={d:[] for d in range(10)}
    for s in scales:
        for t in thicks:
            for d in range(10):
                canvas=np.zeros((32,24),np.uint8)
                cv2.putText(canvas,str(d),(2,26),cv2.FONT_HERSHEY_SIMPLEX,s,255,t,cv2.LINE_AA)
                x,y,w,h=cv2.boundingRect((canvas>0).astype(np.uint8))
                crop=canvas[y:y+h,x:x+w]
                if crop.size>0: templates[d].append(crop)
    _DIGIT_TEMPLATES=templates
    return templates

def _match_digit(glyph: np.ndarray):
    tmpls=_make_digit_templates()
    if glyph is None or glyph.size==0: return 0,-1.0
    g=cv2.GaussianBlur(glyph,(3,3),0)
    g=cv2.normalize(g,None,0,255,cv2.NORM_MINMAX)
    best_d,best=-1,-1.0
    for d,vars in tmpls.items():
        for t in vars:
            th,tw=t.shape[:2]
            Gh,Gw=g.shape[:2]
            if min(Gh,Gw)<=0: continue
            Gs=cv2.resize(g,(tw,th),interpolation=cv2.INTER_AREA)
            res=cv2.matchTemplate(Gs,t,cv2.TM_CCOEFF_NORMED)
            score=float(res.max()) if res.size else -1.0
            if score>best: best, best_d = score, d
    return best_d,best

def _read_weight_near(img_bgr, pt):
    H,W=img_bgr.shape[:2]; x,y=int(pt[0]),int(pt[1])
    sz=max(24,int(0.06*max(H,W)))
    x0,y0=max(0,x-sz),max(0,y-sz); x1,y1=min(W,x+sz),min(H,y+sz)
    roi=img_bgr[y0:y1,x0:x1]
    if roi.size==0: return -1
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV); v=hsv[...,2]
    mask=(v>60).astype(np.uint8)*255
    kern=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kern,iterations=1)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in cnts:
        x2,y2,w2,h2=cv2.boundingRect(c)
        if 8<=w2<=40 and 12<=h2<=45: boxes.append((x2,y2,w2,h2))
    if not boxes: return -1
    boxes.sort(key=lambda b:b[0])
    digits=[]
    for bx,by,bw,bh in boxes:
        d,score=_match_digit(cv2.cvtColor(roi[by:by+bh,bx:bx+bw],cv2.COLOR_BGR2GRAY))
        if score>=0.35: digits.append(str(d))
    if not digits: return -1
    try: return int("".join(digits))
    except: return -1

def _build_graph(img_bgr):
    nodes=_detect_nodes(img_bgr)
    if len(nodes)<2: return 2,[(0,1,1)]
    segs=_detect_line_segments(img_bgr)
    pairs=_segments_to_edges(nodes,segs,img_bgr.shape)
    centers=[(x,y) for x,y,_ in nodes]
    edges=[]
    for i,j,mid in pairs:
        w=_read_weight_near(img_bgr,mid)
        if w>0: edges.append((i,j,int(w)))
    if len(edges)<max(1,len(nodes)-1):
        tried={(min(u,v),max(u,v)) for (u,v,_) in edges}
        for i in range(len(centers)):
            for j in range(i+1,len(centers)):
                if (i,j) in tried: continue
                mid=((centers[i][0]+centers[j][0])//2,(centers[i][1]+centers[j][1])//2)
                w=_read_weight_near(img_bgr,mid)
                if w>0: edges.append((i,j,int(w)))
    best={}
    for u,v,w in edges:
        a,b=(u,v) if u<v else (v,u)
        if (a,b) not in best or w<best[(a,b)]: best[(a,b)]=w
    dedup=[(a,b,w) for (a,b),w in best.items()]
    return len(nodes), dedup

def _solve_single(img_b64: str) -> int:
    img=_b64_to_cv(img_b64)
    n,edges=_build_graph(img)
    return _mst_weight(n,edges)

@app.get("/healthz")
def healthz(): return "ok", 200

@app.post("/mst-calculation")
def mst_calculation():
    data=request.get_json(silent=True)
    if not isinstance(data,list) or not data:
        return jsonify({"error":"Body must be a JSON array of {image} objects"}),400
    out=[]
    for item in data:
        b64=(item or {}).get("image")
        if not isinstance(b64,str):
            return jsonify({"error":"Each item must include an 'image' base64 string"}),400
        try: val=_solve_single(b64)
        except Exception: val=0
        out.append({"value":int(val)})
    return jsonify(out),200

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))





