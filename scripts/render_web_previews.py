"""Render genuine segmented STL surfaces as static previews for non-WebGL views."""
from pathlib import Path
import math
import struct
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'assets' / 'previews'
OUT.mkdir(parents=True, exist_ok=True)

def unit(v):
    length = math.sqrt(sum(x*x for x in v)) or 1
    return tuple(x/length for x in v)

def dot(a, b):
    return sum(x*y for x,y in zip(a,b))

def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def render(src, dst, size=850):
    faces=[];lo=[float('inf')]*3;hi=[float('-inf')]*3
    with src.open('rb') as f:
        f.seek(80); count=struct.unpack('<I',f.read(4))[0]
        for _ in range(count):
            data=f.read(50)
            if len(data)<50:break
            row=struct.unpack('<12fH',data)
            pts=[row[i:i+3] for i in (3,6,9)]
            faces.append(pts)
            for p in pts:
                for k in range(3):lo[k]=min(lo[k],p[k]);hi[k]=max(hi[k],p[k])
    center=tuple((lo[k]+hi[k])/2 for k in range(3))
    view=unit((.92,-1.10,.95));right=unit(cross((0,0,1),view));up=unit(cross(view,right));light=unit((.6,-.4,1))
    scale=size*.72/max(max(hi[k]-lo[k] for k in range(3)),1)
    polygons=[]
    for face in faces:
        pts=[tuple(p[k]-center[k] for k in range(3)) for p in face]
        a=tuple(pts[1][k]-pts[0][k] for k in range(3));b=tuple(pts[2][k]-pts[0][k] for k in range(3))
        normal=unit(cross(a,b))
        if dot(normal,view)<0:normal=tuple(-x for x in normal)
        shade=.35+.58*max(0,dot(normal,light))+.18*(1-max(0,dot(normal,view)))
        color=tuple(min(255,int(c*shade)) for c in (238,164,161))+(255,)
        xy=[(size/2+dot(p,right)*scale,size/2-dot(p,up)*scale) for p in pts]
        polygons.append((sum(dot(p,view) for p in pts)/3,xy,color))
    polygons.sort(key=lambda x:x[0])
    image=Image.new('RGBA',(size,size),(0,0,0,0));draw=ImageDraw.Draw(image)
    for _,pts,color in polygons:draw.polygon(pts,fill=color)
    image.save(dst,optimize=True)

for model in sorted((ROOT/'docs/assets/models').glob('train_*.stl')):
    render(model,OUT/(model.stem+'.png'))
