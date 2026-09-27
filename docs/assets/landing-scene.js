import * as THREE from 'three';
import {STLLoader} from 'three/addons/loaders/STLLoader.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {mergeVertices} from 'three/addons/utils/BufferGeometryUtils.js';
const box=document.getElementById('hero-instrument'),canvas=document.getElementById('valve-canvas');
const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
try {
  const renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.1;
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(40,1,.1,2000);
  scene.add(new THREE.HemisphereLight(0xfff2e8,0x33516a,2));
  const key=new THREE.DirectionalLight(0xffdac9,2.4);key.position.set(1,2,2);scene.add(key);
  const rim=new THREE.DirectionalLight(0x83d9df,1.8);rim.position.set(-2,-1,-2);scene.add(rim);
  const controls=new OrbitControls(camera,canvas);controls.enableDamping=true;controls.autoRotate=!reduced;controls.autoRotateSpeed=.55;controls.enablePan=false;controls.enableZoom=false;
  let mesh, flow, radius=30, view='tissue', visible=true;
  const flowPoints=new Float32Array(160*3),phases=Array.from({length:160},(_,i)=>(i*.61803398875)%1);
  new IntersectionObserver(entries=>visible=entries[0].isIntersecting).observe(box);
  new ResizeObserver(()=>{const w=box.clientWidth,h=box.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}).observe(box);
  new STLLoader().load('segmented_valve_mesh_smoothed.stl',raw=>{
    raw.deleteAttribute('normal');const geo=mergeVertices(raw,.001);raw.dispose();geo.computeVertexNormals();geo.center();geo.computeBoundingSphere();radius=geo.boundingSphere.radius;
    camera.position.set(radius*.8,-radius*2.2,radius*2.2);controls.minDistance=radius*1.8;controls.maxDistance=radius*5;
    mesh=new THREE.Mesh(geo,new THREE.MeshPhysicalMaterial({color:0xc88c83,side:THREE.DoubleSide,roughness:.58,clearcoat:.18}));scene.add(mesh);
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(flowPoints,3));flow=new THREE.Points(g,new THREE.PointsMaterial({color:0x70eee0,size:1.2,transparent:true,opacity:.9}));flow.visible=view==='flow';scene.add(flow);box.classList.add('ready');
  },undefined,()=>{document.getElementById('scene-status').textContent='STATIC PREVIEW';});
  document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{
    view=button.dataset.view;document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',b===button));
    document.getElementById('scene-status').textContent=view==='flow'?'ILLUSTRATIVE FLOW':view.toUpperCase()+' VIEW';
    if(mesh){mesh.material.wireframe=view==='mesh';mesh.material.transparent=view==='flow';mesh.material.opacity=view==='flow'?.32:1;mesh.material.needsUpdate=true;flow.visible=view==='flow';}
  }));
  let last=0;
  renderer.setAnimationLoop(time=>{
    const dt=Math.min((time-last)/1000,.05);last=time;if(!visible||document.hidden)return;
    if(flow&&flow.visible){for(let i=0;i<160;i++){if(!reduced)phases[i]=(phases[i]+dt*.3)%1;const z=(phases[i]-.5)*radius*2.2,a=i*2.399963,width=radius*(.07+.14*Math.abs(z/radius));flowPoints[i*3]=Math.cos(a)*width;flowPoints[i*3+1]=Math.sin(a)*width;flowPoints[i*3+2]=z;}flow.geometry.attributes.position.needsUpdate=true;}
    controls.update();renderer.render(scene,camera);
  });
} catch(error) { document.getElementById('scene-status').textContent='MESH PREVIEW'; }
