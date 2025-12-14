---
title: "Probably Approximately Correct"
---

An ML engineer's musings on machine learning and life...
{.lead}

<div class="arcade" id="pongArcade">
  <div class="arcade__header">
    <div class="arcade__title">no pongs intended</div>
    <div class="arcade__hint">
      🏓 click/tap to begin • ⏸️ pause: space key or button • 🔄 restart: R key or button
    </div>
  </div>

  <canvas id="pongCanvas" class="arcade__canvas" width="720" height="420"></canvas>

  <!-- mobile + desktop controls -->
  <div class="arcade__controls" aria-label="pong controls">
    <button class="arcade__btn" id="pongStartBtn" type="button">start</button>
    <button class="arcade__btn" id="pongPauseBtn" type="button">pause</button>
    <button class="arcade__btn" id="pongRestartBtn" type="button">restart</button>
  </div>

  <div class="arcade__meta">
    <span class="arcade__pill">will</span>
    <span id="pongLeft" class="arcade__score">0</span>
    <span class="arcade__sep">:</span>
    <span id="pongRight" class="arcade__score">0</span>
    <span class="arcade__pill">vecna</span>
    <span id="pongMsg" class="arcade__msg">click/tap start</span>
  </div>
</div>

<script>
document.addEventListener("DOMContentLoaded", function () {
  const canvas = document.getElementById("pongCanvas");
  const ctx = canvas.getContext("2d");

  const leftScoreEl = document.getElementById("pongLeft");
  const rightScoreEl = document.getElementById("pongRight");
  const msgEl = document.getElementById("pongMsg");

  const startBtn = document.getElementById("pongStartBtn");
  const pauseBtn = document.getElementById("pongPauseBtn");
  const restartBtn = document.getElementById("pongRestartBtn");

  const W = canvas.width;
  const H = canvas.height;

  // vibe knobs (stranger-things: black board, white paddles/line, red glowing ball)
  const RED = "#ff2d55";
  const WHITE = "#ffffff";
  const WHITE_SOFT = "rgba(255,255,255,0.22)";
  const BG = "#000000";

  // game constants
  const paddleW = 10;
  const paddleH = 84;
  const ballR = 6;

  const leftX = 22;
  const rightX = W - 22 - paddleW;

  let leftY, rightY;
  let ballX, ballY, ballVX, ballVY;

  let leftScore = 0, rightScore = 0;
  let started = false;
  let paused = false;

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function resetRound(direction) {
    ballX = W / 2;
    ballY = H / 2;

    const speed = 6.2;
    const angle = (Math.random() * 0.8 - 0.4);
    ballVX = speed * (direction || (Math.random() < 0.5 ? -1 : 1));
    ballVY = speed * angle;

    leftY = clamp(leftY ?? (H - paddleH) / 2, 0, H - paddleH);
    rightY = clamp(rightY ?? (H - paddleH) / 2, 0, H - paddleH);

    msgEl.textContent = started ? "" : "click/tap start";
  }

  function resetGame() {
    leftScore = 0; rightScore = 0;
    leftScoreEl.textContent = "0";
    rightScoreEl.textContent = "0";
    started = false;
    paused = false;
    leftY = (H - paddleH) / 2;
    rightY = (H - paddleH) / 2;
    resetRound();
    draw();
  }

  function inView() {
    const r = canvas.getBoundingClientRect();
    return r.bottom > 0 && r.top < window.innerHeight;
  }

  function startGame() {
    if (!started) {
      started = true;
      paused = false;
      msgEl.textContent = "";
      resetRound(Math.random() < 0.5 ? -1 : 1);
      draw();
    }
  }

  function togglePause() {
    if (!started) {
      msgEl.textContent = "click/tap start";
      return;
    }
    paused = !paused;
    msgEl.textContent = paused ? "paused" : "";
    draw();
  }

  // mouse (and touch) controls left paddle
  function setLeftFromClientY(clientY) {
    const r = canvas.getBoundingClientRect();
    const y = (clientY - r.top) * (H / r.height) - paddleH / 2;
    leftY = clamp(y, 0, H - paddleH);
  }

  canvas.addEventListener("mousemove", (e) => setLeftFromClientY(e.clientY));

  // touch drag to move paddle (also prevents page scroll while dragging on canvas)
  canvas.addEventListener("touchmove", (e) => {
    if (!e.touches.length) return;
    setLeftFromClientY(e.touches[0].clientY);
    e.preventDefault();
  }, { passive: false });

  // click/tap to start (works on desktop + mobile)
  canvas.addEventListener("mousedown", startGame);
  canvas.addEventListener("touchstart", (e) => {
    if (!inView()) return;
    startGame();
    // prevent "ghost click" on some mobile browsers
    e.preventDefault();
  }, { passive: false });

  // button controls (mobile + desktop)
  startBtn.addEventListener("click", startGame);
  pauseBtn.addEventListener("click", togglePause);
  restartBtn.addEventListener("click", resetGame);

  // keyboard helpers (desktop)
  window.addEventListener("keydown", (e) => {
    if (!inView()) return;
    const k = e.key.toLowerCase();
    if (k === " ") {
      togglePause();
      e.preventDefault();
    } else if (k === "r") {
      resetGame();
      e.preventDefault();
    }
  }, { passive: false });

  function drawBackground() {
    // solid black board
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, W, H);

    // center dashed line (white)
    ctx.fillStyle = WHITE_SOFT;
    const dashH = 18, gap = 12, dashW = 6;
    for (let y = 20; y < H - 20; y += dashH + gap) {
      ctx.fillRect(W / 2 - dashW / 2, y, dashW, dashH);
    }
  }

  function drawPaddlesAndBall() {
    // paddles: white
    ctx.fillStyle = WHITE;
    ctx.fillRect(leftX, leftY, paddleW, paddleH);
    ctx.fillRect(rightX, rightY, paddleW, paddleH);

    // ball: red glow
    ctx.fillStyle = RED;

    // glow pass
    ctx.shadowColor = "rgba(255,45,85,0.75)";
    ctx.shadowBlur = 22;
    ctx.beginPath();
    ctx.arc(ballX, ballY, ballR, 0, Math.PI * 2);
    ctx.fill();

    // crisp core
    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.arc(ballX, ballY, ballR, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawOverlay(text) {
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(0, H / 2 - 34, W, 68);

    ctx.fillStyle = "rgba(255,255,255,0.92)";
    ctx.font = "600 18px system-ui, -apple-system, Segoe UI, Roboto, Arial";
    ctx.textAlign = "center";
    ctx.fillText(text, W / 2, H / 2 + 6);
  }

  function draw() {
    drawBackground();
    drawPaddlesAndBall();

    if (!started) drawOverlay("click/tap start");
    else if (paused) drawOverlay("paused — tap pause / press space");
  }

  function cpuStep() {
    const target = ballY - paddleH / 2;
    const maxSpeed = 5.2;

    const desired = (ballVX > 0) ? target : (H - paddleH) / 2;

    const dy = desired - rightY;
    rightY += clamp(dy, -maxSpeed, maxSpeed);
    rightY = clamp(rightY, 0, H - paddleH);
  }

  function collidePaddle(px, py) {
    const withinX = ballX + ballR >= px && ballX - ballR <= px + paddleW;
    const withinY = ballY + ballR >= py && ballY - ballR <= py + paddleH;
    if (!withinX || !withinY) return false;

    const paddleCenter = py + paddleH / 2;
    const offset = (ballY - paddleCenter) / (paddleH / 2);
    const speedUp = 1.03;

    ballVX = -ballVX * speedUp;
    ballVY = ballVY + offset * 1.8;
    ballVY = clamp(ballVY, -8.5, 8.5);

    if (ballVX < 0) ballX = px - ballR - 0.1;
    else ballX = px + paddleW + ballR + 0.1;

    return true;
  }

  function step() {
    if (!started || paused) return;

    ballX += ballVX;
    ballY += ballVY;

    if (ballY - ballR < 0) { ballY = ballR; ballVY = -ballVY; }
    if (ballY + ballR > H) { ballY = H - ballR; ballVY = -ballVY; }

    collidePaddle(leftX, leftY);
    collidePaddle(rightX, rightY);

    if (ballX + ballR < 0) {
      rightScore += 1;
      rightScoreEl.textContent = String(rightScore);
      resetRound(1);
    } else if (ballX - ballR > W) {
      leftScore += 1;
      leftScoreEl.textContent = String(leftScore);
      resetRound(-1);
    }

    cpuStep();
  }

  let last = 0;
  function loop(t) {
    if (!last) last = t;
    const dt = t - last;
    if (dt >= 16) {
      last = t;
      step();
      draw();
    }
    requestAnimationFrame(loop);
  }

  resetGame();
  requestAnimationFrame(loop);
});
</script>

<style>
/* hawkins lab frame glow */
.arcade {
  max-width: 760px;
  margin: 2.25rem auto 2.5rem;
  padding: 1rem 1rem 0.95rem;

  border-radius: 14px;
  background: rgba(0, 0, 0, 0.92);

  border: 1px solid rgba(255, 45, 85, 0.35);
  box-shadow:
    0 0 18px rgba(255, 45, 85, 0.22),
    0 0 48px rgba(255, 45, 85, 0.14);
}

.arcade__header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 0.65rem;
  flex-wrap: wrap;
}

.arcade__title {
  font-weight: 750;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  font-size: 0.85rem;
  color: rgba(255,255,255,0.85);
}

.arcade__hint {
  font-size: 0.82rem;
  color: rgba(255,255,255,0.55);
}

.arcade__canvas {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.12);
  background: #000;
  touch-action: none; /* important for mobile dragging */
}

/* mobile + desktop buttons */
.arcade__controls {
  margin-top: 0.75rem;
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
}

.arcade__btn {
  appearance: none;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.82);
  padding: 0.45rem 0.7rem;
  border-radius: 10px;
  font-size: 0.9rem;
  letter-spacing: 0.03em;
  cursor: pointer;
}

.arcade__btn:hover {
  border-color: rgba(255,45,85,0.45);
  box-shadow: 0 0 18px rgba(255,45,85,0.18);
}

.arcade__btn:active {
  transform: translateY(1px);
}

.arcade__meta {
  margin-top: 0.75rem;
  display: flex;
  gap: 0.6rem;
  align-items: center;
  flex-wrap: wrap;
  color: rgba(255,255,255,0.70);
  font-size: 0.95rem;
}

.arcade__pill {
  font-size: 0.78rem;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  padding: 0.18rem 0.5rem;
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 999px;
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.78);
}

.arcade__score {
  color: #ff2d55;
  font-weight: 800;
  font-variant-numeric: tabular-nums;
}

.arcade__sep {
  color: rgba(255,255,255,0.35);
  font-weight: 700;
}

.arcade__msg {
  margin-left: 0.35rem;
  color: rgba(255,255,255,0.45);
}

@media (max-width: 520px) {
  .arcade { padding: 0.9rem 0.85rem 0.85rem; }
  .arcade__hint { font-size: 0.78rem; }
  .arcade__btn { padding: 0.5rem 0.75rem; }
}
</style>
