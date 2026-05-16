// esp8266/car_firmware.ino
// ─────────────────────────────────────────────────────────────
// ESP8266 — AP mode + HTTP server + WebSocket server
//
// Port 80 : HTTP — serves the gesture control web app
// Port 81 : WebSocket — receives motor commands
//
// User connects phone to GestureCar WiFi
// Opens http://192.168.4.1 in phone browser
// App loads, connects WebSocket, controls car
//
// Required libraries (Arduino Library Manager):
//   WebSockets  by Markus Sattler
//   ESP8266WebServer  (built into ESP8266 board package)
// ─────────────────────────────────────────────────────────────

#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <WebSocketsServer.h>

// ── AP credentials ─────────────────────────────────────────────
const char* AP_SSID = "GestureCar";
const char* AP_PASS = "";           // open network

// ── Servers ────────────────────────────────────────────────────
ESP8266WebServer http(80);
WebSocketsServer ws(81);

// ── Motor pins ─────────────────────────────────────────────────
#define IN1  D1
#define IN2  D2
#define IN3  D5
#define IN4  D6
#define ENA  D7
#define ENB  D8
#define SPEED 700

// ── State ──────────────────────────────────────────────────────
int  activeClient = -1;
unsigned long lastCmd   = 0;
const unsigned long WATCHDOG_MS = 600;

// ── Control app HTML ───────────────────────────────────────────
// Stored in flash (PROGMEM) — does not use RAM
// This is the full gesture control UI served to the phone
// MediaPipe + ONNX loaded from CDN (phone needs internet
// for first load, then cached)
//
// NOTE: model.onnx and labels.json are served from GitHub Pages
// and fetched by the app on first load then cached by browser.
// The ONNX_BASE_URL points to your GitHub Pages deployment.
// Update this URL after you deploy to GitHub Pages.

const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no"/>
<title>Gesture RC Car</title>
<style>
:root{
  --bg:#0f1117;--surface:#1a1d27;--border:#2a2d3a;
  --text:#e8eaf0;--muted:#6b7080;
  --green:#00e676;--blue:#4dabf7;--amber:#ffb300;
  --red:#ff5252;--purple:#b39ddb;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{
  font-family:'Segoe UI',system-ui,sans-serif;
  background:var(--bg);color:var(--text);
  min-height:100dvh;display:flex;flex-direction:column;
  -webkit-tap-highlight-color:transparent;overflow-x:hidden;
}
.overlay{
  position:fixed;inset:0;background:rgba(15,17,23,.97);
  display:flex;flex-direction:column;align-items:center;
  justify-content:center;z-index:900;padding:28px;
  text-align:center;gap:16px;
}
.overlay.gone{display:none;}
#load-overlay{z-index:999;}
.spinner{
  width:48px;height:48px;border:3px solid var(--border);
  border-top-color:var(--green);border-radius:50%;
  animation:spin 1s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg);}}
.overlay h2{font-size:22px;font-weight:800;}
.overlay p{font-size:14px;color:var(--muted);line-height:1.8;max-width:320px;}
#busy-overlay .busy-icon{font-size:52px;}
#busy-overlay h2{color:var(--amber);}
.btn-primary{
  background:var(--green);border:none;border-radius:12px;
  padding:15px 32px;color:#000;font-size:16px;font-weight:800;
  cursor:pointer;width:100%;max-width:340px;
}
.btn-primary:active{transform:scale(.97);}
header{
  background:var(--surface);border-bottom:1px solid var(--border);
  padding:11px 16px;display:flex;justify-content:space-between;
  align-items:center;position:sticky;top:0;z-index:100;
}
header h1{font-size:16px;font-weight:700;}
.pills{display:flex;gap:6px;}
.pill{
  font-size:11px;padding:4px 10px;border-radius:20px;
  border:1px solid var(--border);color:var(--muted);
  background:var(--bg);transition:all .25s;white-space:nowrap;
}
.pill.on{color:var(--green);border-color:var(--green);}
.pill.err{color:var(--red);border-color:var(--red);}
.pill.warn{color:var(--amber);border-color:var(--amber);}
main{flex:1;display:flex;flex-direction:column;gap:12px;padding:12px;}
.cam-wrap{
  position:relative;background:#000;border-radius:12px;
  overflow:hidden;aspect-ratio:4/3;
}
#cam-canvas{width:100%;height:100%;object-fit:cover;
  display:block;transform:scaleX(-1);}
.cam-hud{
  position:absolute;inset:0;display:flex;
  flex-direction:column;justify-content:space-between;
  padding:10px;pointer-events:none;
}
.hud-top{display:flex;justify-content:space-between;}
.g-badge{
  background:rgba(0,0,0,.75);padding:5px 13px;
  border-radius:20px;font-size:14px;font-weight:600;color:var(--green);
}
.fps{background:rgba(0,0,0,.6);padding:3px 8px;
  border-radius:6px;font-size:11px;color:var(--muted);}
.conf-row{display:flex;align-items:center;gap:8px;}
.conf-track{flex:1;height:5px;background:rgba(255,255,255,.15);
  border-radius:3px;overflow:hidden;}
.conf-fill{height:100%;border-radius:3px;
  background:var(--green);width:0;transition:width .12s;}
.conf-pct{font-size:11px;color:var(--muted);min-width:30px;text-align:right;}
.card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:14px;
}
.card-label{font-size:11px;color:var(--muted);
  text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px;}
.cmd{
  font-size:54px;font-weight:900;letter-spacing:3px;
  text-align:center;padding:8px 0;color:var(--red);transition:color .12s;
}
.cmd.FORWARD{color:var(--green);}
.cmd.REVERSE{color:var(--blue);}
.cmd.LEFT{color:var(--purple);}
.cmd.RIGHT{color:var(--amber);}
.dpad{display:flex;flex-direction:column;align-items:center;gap:6px;}
.dpad-row{display:flex;gap:6px;}
.dpad-btn{
  width:62px;height:62px;border-radius:10px;
  background:var(--bg);border:1px solid var(--border);
  display:flex;align-items:center;justify-content:center;
  font-size:22px;color:var(--muted);transition:all .1s;
}
.dpad-btn.lit{
  background:#0d2b1a;border-color:var(--green);
  color:var(--green);box-shadow:0 0 16px rgba(0,230,118,.35);
}
.dpad-btn.lit.stp{
  background:#2a0d0d;border-color:var(--red);
  color:var(--red);box-shadow:0 0 16px rgba(255,82,82,.35);
}
.conn-bar{
  display:flex;align-items:center;gap:10px;
  padding:9px 12px;background:var(--bg);
  border-radius:8px;border:1px solid var(--border);
}
.dot{width:8px;height:8px;border-radius:50%;
  background:var(--muted);flex-shrink:0;}
.dot.on{background:var(--green);box-shadow:0 0 6px var(--green);}
.dot.err{background:var(--red);}
.conn-text{flex:1;font-size:12px;color:var(--muted);}
.start-btn{
  background:var(--green);border:none;border-radius:12px;
  padding:17px;color:#000;font-size:17px;font-weight:800;
  width:100%;cursor:pointer;transition:all .15s;
}
.start-btn.running{background:var(--red);color:#fff;}
.start-btn:active{transform:scale(.97);}
@media(min-width:580px){
  main{flex-direction:row;flex-wrap:wrap;}
  .cam-wrap{max-width:380px;}
  .right{flex:1;display:flex;flex-direction:column;
    gap:12px;min-width:260px;}
}
</style>
</head>
<body>

<div class="overlay" id="load-overlay">
  <div class="spinner"></div>
  <h2>Gesture RC Car</h2>
  <p id="load-txt">Loading ML model&hellip;<br>
    <small>First load ~15s &middot; cached after</small></p>
</div>

<div class="overlay gone" id="busy-overlay">
  <div class="busy-icon">&#x1F512;</div>
  <h2>Car Busy</h2>
  <p>Already controlled by another device.<br>
     Ask them to close the app, then retry.</p>
  <button class="btn-primary" id="busy-retry">&#x21BA; Try Again</button>
</div>

<header>
  <h1>&#x2B21; Gesture RC Car</h1>
  <div class="pills">
    <span class="pill" id="p-cam">&#9679; Cam</span>
    <span class="pill" id="p-ml">&#9679; ML</span>
    <span class="pill" id="p-car">&#9679; Car</span>
  </div>
</header>

<main>
  <div class="cam-wrap">
    <canvas id="cam-canvas"></canvas>
    <div class="cam-hud">
      <div class="hud-top">
        <div class="g-badge" id="g-badge">&mdash;</div>
        <div class="fps" id="fps-el">0 fps</div>
      </div>
      <div class="conf-row">
        <div class="conf-track">
          <div class="conf-fill" id="conf-fill"></div>
        </div>
        <span class="conf-pct" id="conf-pct">0%</span>
      </div>
    </div>
  </div>

  <div class="right">
    <div class="card">
      <div class="card-label">Active Command</div>
      <div class="cmd STOP" id="cmd-el">STOP</div>
    </div>
    <div class="card">
      <div class="dpad">
        <div class="dpad-btn" id="d-FORWARD">&#9650;</div>
        <div class="dpad-row">
          <div class="dpad-btn" id="d-LEFT">&#9664;</div>
          <div class="dpad-btn stp" id="d-STOP">&#9632;</div>
          <div class="dpad-btn" id="d-RIGHT">&#9654;</div>
        </div>
        <div class="dpad-btn" id="d-REVERSE">&#9660;</div>
      </div>
    </div>
    <div class="card">
      <div class="card-label">Car Connection</div>
      <div class="conn-bar">
        <div class="dot" id="conn-dot"></div>
        <span class="conn-text" id="conn-txt">Connecting&hellip;</span>
      </div>
    </div>
    <button class="start-btn" id="start-btn">&#9654; Start Gesture Control</button>
  </div>
</main>

<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js" crossorigin="anonymous"></script>
<script>
// ── Config ──────────────────────────────────────────────────────
// WebSocket on same host — HTTP page so ws:// is allowed
const CAR_WS = 'ws://' + location.hostname + ':81';

// model.onnx and labels.json served from GitHub Pages
// UPDATE THIS to your actual GitHub Pages URL after deploying
const ASSETS_BASE = 'https://yash-patil-26.github.io/Gesture_Car';

const CONF_MIN = 0.85;
const VOTES    = 3;

// ── State ───────────────────────────────────────────────────────
let session, labels=[], ws;
let running=false, isBusy=false;
let voteBuf=[], lastStable='STOP';
let lastCmd=null, lastSent=0;
let fpsCnt=0, fpsVal=0, fpsT=performance.now();
let camera, hands, retryTimer, retryN=0;

// ── DOM ─────────────────────────────────────────────────────────
const canvas  = document.getElementById('cam-canvas');
const ctx     = canvas.getContext('2d');
const gBadge  = document.getElementById('g-badge');
const fpsEl   = document.getElementById('fps-el');
const cFill   = document.getElementById('conf-fill');
const cPct    = document.getElementById('conf-pct');
const cmdEl   = document.getElementById('cmd-el');
const connDot = document.getElementById('conn-dot');
const connTxt = document.getElementById('conn-txt');
const startBtn= document.getElementById('start-btn');
const pCam    = document.getElementById('p-cam');
const pMl     = document.getElementById('p-ml');
const pCar    = document.getElementById('p-car');
const DPAD    = ['FORWARD','REVERSE','LEFT','RIGHT','STOP']
  .reduce((o,k)=>(o[k]=document.getElementById('d-'+k),o),{});

// ── Overlays ────────────────────────────────────────────────────
const hide = id => document.getElementById(id).classList.add('gone');
const show = id => document.getElementById(id).classList.remove('gone');
const hideAll = () => ['load-overlay','busy-overlay'].forEach(hide);

document.getElementById('busy-retry').addEventListener('click',()=>{
  isBusy=false; hide('busy-overlay'); retryN=0; connect();
});

// ── Connection ──────────────────────────────────────────────────
function setConn(state, msg) {
  const s = {ok:'on',busy:'err',off:'err',wait:''};
  connDot.className = 'dot' + (s[state] ? ' '+s[state] : '');
  connTxt.textContent = msg;
  pCar.className = 'pill' + (s[state] ? ' '+s[state] : '');
}

function connect() {
  if (isBusy) return;
  if (retryTimer) { clearTimeout(retryTimer); retryTimer=null; }
  if (ws) { try{ws.close();}catch(_){} ws=null; }
  setConn('wait','Connecting to car…');
  try { ws = new WebSocket(CAR_WS); } catch(e) { retry(); return; }

  ws.onopen  = () => { retryN=0; setConn('ok','Car connected ✓'); };
  ws.onclose = () => { if(!isBusy){ setConn('off','Car disconnected'); retry(); } };
  ws.onerror = () => setConn('off','Connection error');
  ws.onmessage = e => {
    if (e.data.startsWith('BUSY:')) {
      isBusy=true;
      try{ws.close();}catch(_){}
      setConn('busy','Car busy — another controller active');
      hide('load-overlay'); show('busy-overlay');
    }
  };
}

function retry() {
  retryN++;
  retryTimer = setTimeout(connect, Math.min(2000*retryN, 10000));
}

// ── Send ────────────────────────────────────────────────────────
function send(cmd) {
  const now=Date.now();
  if ((cmd!==lastCmd||now-lastSent>300) &&
      ws?.readyState===WebSocket.OPEN) {
    ws.send(cmd); lastCmd=cmd; lastSent=now;
  }
}

// ── Cleanup ─────────────────────────────────────────────────────
function cleanup() {
  if (ws?.readyState===WebSocket.OPEN) { ws.send('STOP'); ws.close(1000); }
  camera?.stop();
}
window.addEventListener('beforeunload', cleanup);
window.addEventListener('pagehide',     cleanup);
document.addEventListener('visibilitychange',
  () => { if(document.hidden && running) send('STOP'); });

// ── Vote buffer ─────────────────────────────────────────────────
function vote(label, conf, hand) {
  if (!hand||conf<CONF_MIN) { voteBuf=[]; lastStable='STOP'; return 'STOP'; }
  const cmd=label.toUpperCase();
  if (cmd==='STOP') { voteBuf=[]; lastStable='STOP'; return 'STOP'; }
  voteBuf.push(cmd);
  if (voteBuf.length>VOTES) voteBuf.shift();
  if (voteBuf.length>=VOTES && voteBuf.every(v=>v===voteBuf[0]))
    lastStable=voteBuf[0];
  return lastStable;
}

// ── Features ────────────────────────────────────────────────────
function feat(lms) {
  const wx=lms[0].x,wy=lms[0].y,wz=lms[0].z,c=[];
  for(const l of lms) c.push(l.x-wx,l.y-wy,l.z-wz);
  const mx=Math.max(...c.map(Math.abs));
  return mx<1e-6?null:new Float32Array(c.map(v=>v/mx));
}

// ── ONNX ────────────────────────────────────────────────────────
async function classify(f) {
  if (!session) return null;
  try {
    const t=new ort.Tensor('float32',f,[1,63]);
    const r=await session.run({float_input:t});
    // Try both possible output key names from skl2onnx
    const probs = r['probabilities']?.data
               || r['output_probability']?.data
               || r[Object.keys(r).find(k=>k.includes('prob'))]?.data;
    if (!probs) return null;
    let mi=0,mp=0;
    probs.forEach((v,i)=>{ if(v>mp){mp=v;mi=i;} });
    return {label:labels[mi]||'?', conf:mp};
  } catch { return null; }
}

// ── Draw landmarks ───────────────────────────────────────────────
const CONNS=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]];
const TIPS=new Set([4,8,12,16,20]);

function drawLandmarks(lms) {
  const w=canvas.width,h=canvas.height;
  ctx.strokeStyle='rgba(255,255,255,.45)';ctx.lineWidth=1.5;
  for(const[a,b]of CONNS){
    ctx.beginPath();
    ctx.moveTo(lms[a].x*w,lms[a].y*h);
    ctx.lineTo(lms[b].x*w,lms[b].y*h);
    ctx.stroke();
  }
  lms.forEach((l,i)=>{
    ctx.beginPath();
    ctx.arc(l.x*w,l.y*h,i===0?6:TIPS.has(i)?5:3,0,Math.PI*2);
    ctx.fillStyle=i===0?'#ffb300':TIPS.has(i)?'#00e676':'#4dabf7';
    ctx.fill();
  });
}

// ── MediaPipe result ─────────────────────────────────────────────
async function onResults(res) {
  canvas.width=res.image.width;canvas.height=res.image.height;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(res.image,0,0);
  let g='No hand',c=0,hand=false;
  if (res.multiHandLandmarks?.length) {
    const lms=res.multiHandLandmarks[0];
    hand=true; drawLandmarks(lms);
    const f=feat(lms);
    if(f){ const r=await classify(f); if(r){g=r.label;c=r.conf;} }
  }
  const cmd=vote(g,c,hand);
  gBadge.textContent=g;
  const pct=Math.round(c*100);
  cFill.style.width=pct+'%';
  cFill.style.background=pct>=85?'#00e676':'#4dabf7';
  cPct.textContent=pct+'%';
  cmdEl.textContent=cmd; cmdEl.className='cmd '+cmd;
  Object.entries(DPAD).forEach(([k,el])=>
    el.classList.toggle('lit',k===cmd));
  fpsCnt++;
  const now=performance.now();
  if(now-fpsT>=1000){fpsVal=fpsCnt;fpsCnt=0;fpsT=now;
    fpsEl.textContent=fpsVal+' fps';}
  if(running) send(cmd);
}

// ── Camera ───────────────────────────────────────────────────────
async function startCamera() {
  if (!hands) {
    hands=new Hands({
      locateFile:f=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
    });
    hands.setOptions({maxNumHands:1,modelComplexity:1,
      minDetectionConfidence:.7,minTrackingConfidence:.6});
    hands.onResults(onResults);
  }
  try {
    const stream=await navigator.mediaDevices.getUserMedia(
      {video:{facingMode:'user',width:{ideal:640},height:{ideal:480}}});
    const vid=document.createElement('video');
    vid.srcObject=stream;
    camera=new Camera(vid,{
      onFrame:async()=>{await hands.send({image:vid});},
      width:640,height:480
    });
    await camera.start();
    pCam.classList.add('on');
  } catch {
    pCam.classList.add('err');
    alert('Camera access denied — allow camera in browser settings.');
  }
}

// ── Start button ─────────────────────────────────────────────────
startBtn.addEventListener('click',async()=>{
  if (!running) {
    running=true;
    startBtn.textContent='&#9209; Stop';
    startBtn.classList.add('running');
    await startCamera();
  } else {
    running=false;
    startBtn.textContent='&#9654; Start Gesture Control';
    startBtn.classList.remove('running');
    send('STOP'); camera?.stop(); camera=null;
    cmdEl.textContent='STOP'; cmdEl.className='cmd STOP';
    Object.values(DPAD).forEach(el=>el.classList.remove('lit'));
  }
});

// ── Load model ───────────────────────────────────────────────────
async function loadModel() {
  const ltxt=document.getElementById('load-txt');
  try {
    ltxt.innerHTML='Downloading ML model&hellip;<br><small>~10MB &middot; cached after first load</small>';
    // Fetch model from GitHub Pages (CORS allowed for same-origin static files)
    session=await ort.InferenceSession.create(
      ASSETS_BASE+'/model.onnx',
      {executionProviders:['wasm']}
    );
    ltxt.innerHTML='Loading labels&hellip;';
    labels=(await(await fetch(ASSETS_BASE+'/labels.json')).json()).labels;
    hideAll();
    pMl.classList.add('on');
    connect();
  } catch(e) {
    ltxt.innerHTML='Load failed.<br><small>'+e.message+'</small>';
    pMl.classList.add('err');
  }
}

loadModel();
</script>
</body>
</html>
)rawliteral";

// ── Motors ──────────────────────────────────────────────────────
void stopMotors() {
  digitalWrite(IN1,LOW);digitalWrite(IN2,LOW);
  digitalWrite(IN3,LOW);digitalWrite(IN4,LOW);
  analogWrite(ENA,0);analogWrite(ENB,0);
}
void drive(bool l1,bool l2,bool r1,bool r2,int spd){
  digitalWrite(IN1,l1);digitalWrite(IN2,l2);
  digitalWrite(IN3,r1);digitalWrite(IN4,r2);
  analogWrite(ENA,spd);analogWrite(ENB,spd);
}
void execute(String cmd){
  cmd.trim();cmd.toUpperCase();
  if      (cmd=="FORWARD") drive(1,0,1,0,SPEED);
  else if (cmd=="REVERSE") drive(0,1,0,1,SPEED);
  else if (cmd=="LEFT")    drive(0,1,1,0,SPEED);
  else if (cmd=="RIGHT")   drive(1,0,0,1,SPEED);
  else                     stopMotors();
  lastCmd=millis();
}

// ── WebSocket handler ────────────────────────────────────────────
void onWsEvent(uint8_t num,WStype_t type,
               uint8_t* payload,size_t len){
  switch(type){
    case WStype_CONNECTED:
      if(activeClient==-1){
        activeClient=num;lastCmd=millis();
        ws.sendTXT(num,"READY");
        Serial.printf("[WS] Client #%d active\n",num);
      } else {
        ws.sendTXT(num,
          "BUSY:Car already controlled by another device. "
          "Ask them to close the app first.");
        delay(80);
        ws.disconnect(num);
        Serial.printf("[WS] Client #%d rejected — busy\n",num);
      }
      break;
    case WStype_DISCONNECTED:
      if(num==activeClient){
        activeClient=-1;stopMotors();
        Serial.println("[WS] Controller left — motors stopped");
      }
      break;
    case WStype_TEXT:
      if(num==activeClient){
        execute(String((char*)payload));
      }
      break;
    default:break;
  }
}

// ── Setup ────────────────────────────────────────────────────────
void setup(){
  Serial.begin(115200);delay(100);
  pinMode(IN1,OUTPUT);digitalWrite(IN1,LOW);
  pinMode(IN2,OUTPUT);digitalWrite(IN2,LOW);
  pinMode(IN3,OUTPUT);digitalWrite(IN3,LOW);
  pinMode(IN4,OUTPUT);digitalWrite(IN4,LOW);
  pinMode(ENA,OUTPUT);analogWrite(ENA,0);
  pinMode(ENB,OUTPUT);analogWrite(ENB,0);

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID,AP_PASS);

  // HTTP: serve control app at root
  http.on("/",[](){ 
    http.send_P(200,"text/html",INDEX_HTML);
  });
  http.begin();

  ws.begin();
  ws.onEvent(onWsEvent);
  lastCmd=millis();

  IPAddress ip=WiFi.softAPIP();
  Serial.println("\n═══════════════════════════════════");
  Serial.println("  Gesture RC Car — Ready");
  Serial.printf ("  WiFi   : %s\n",AP_SSID);
  Serial.printf ("  App    : http://%s\n",ip.toString().c_str());
  Serial.printf ("  WS     : ws://%s:81\n",ip.toString().c_str());
  Serial.println("═══════════════════════════════════");
  Serial.println("  1. Connect phone to WiFi above");
  Serial.println("  2. Open http://192.168.4.1 in browser");
  Serial.println("  3. Tap Start — control the car");
  Serial.println("═══════════════════════════════════");
}

// ── Loop ─────────────────────────────────────────────────────────
void loop(){
  http.handleClient();
  ws.loop();
  if(activeClient!=-1 && millis()-lastCmd>WATCHDOG_MS){
    stopMotors();lastCmd=millis();
  }
  yield();
}