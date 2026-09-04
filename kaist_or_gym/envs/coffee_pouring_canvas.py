"""Browser renderer for smooth interpolation between environment keyframes.

The Gymnasium environment owns every scene value through ``render_snapshot``.
This module only draws those values in a browser canvas.  In particular, it
interpolates joint angles and reruns rigid forward kinematics on every display
frame; it never interpolates arm endpoints.
"""

CANVAS_HTML = """
<div class="coffee-stage">
  <canvas class="coffee-canvas" role="img"
    aria-label="Two fixed-link robot arms pouring coffee"></canvas>
  <div class="coffee-render-note">
    <span class="coffee-live-dot"></span>
    continuous motor display · discrete Gymnasium decisions
  </div>
</div>
"""

CANVAS_CSS = """
.coffee-stage {
  width: 100%;
  overflow: hidden;
  border: 1px solid #d8e0e3;
  border-radius: 9px;
  background: #f5f7f8;
  box-shadow: 0 1px 2px rgba(29, 43, 58, 0.08);
}
.coffee-canvas {
  display: block;
  width: 100%;
  aspect-ratio: 12 / 7;
  background: #f5f7f8;
}
.coffee-render-note {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 7px 12px 8px;
  border-top: 1px solid #e1e7e9;
  background: #ffffff;
  color: #52656f;
  font: 600 12px/1.2 ui-sans-serif, system-ui, sans-serif;
  letter-spacing: 0.01em;
}
.coffee-live-dot {
  width: 7px;
  height: 7px;
  flex: 0 0 7px;
  border-radius: 50%;
  background: #2b7a78;
}
"""

CANVAS_JAVASCRIPT = r"""
const LOGICAL_WIDTH = 960;
const LOGICAL_HEIGHT = 560;
const prefersReducedMotion = window.matchMedia(
  "(prefers-reduced-motion: reduce)"
).matches;

let fromState = null;
let targetState = null;
let segmentStart = performance.now();
let segmentDuration = 0;
let lastGeneration = -1;
let lastRevision = -1;
let canvasRef = null;
let resizeObserver = null;
let animationHandle = 0;

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

function mix(a, b, amount) {
  return a + (b - a) * amount;
}

function mixArray(a, b, amount) {
  return a.map((value, index) => mix(value, b[index], amount));
}

function normalizedSnapshot(snapshot) {
  const state = snapshot.state;
  const liquid = state.liquid;
  const playback = snapshot.playback;
  return {
    geometry: snapshot.geometry,
    q: state.joint_angles_rad.slice(),
    fill: liquid.fill_l,
    spill: liquid.spill_l,
    targetFill: liquid.target_fill_l,
    lastFlow: liquid.last_flow_l,
    lastFlowRate: liquid.last_flow_rate_l_s,
    lastCaptured: liquid.last_captured_l,
    streamEnd: liquid.stream_end_m.slice(),
    step: state.step,
    elapsedTime: state.elapsed_time_s,
    dt: state.dt_s,
    horizon: state.horizon_steps,
    terminationReason: state.termination_reason,
    generation: playback.generation,
    revision: playback.revision,
    speed: playback.speed,
    paused: playback.paused,
    running: playback.running,
    intervalMs: playback.decision_interval_wall_ms,
  };
}

function blendedState(first, second, amount) {
  return {
    ...second,
    q: mixArray(first.q, second.q, amount),
    fill: mix(first.fill, second.fill, amount),
    spill: mix(first.spill, second.spill, amount),
    lastFlow: mix(first.lastFlow, second.lastFlow, amount),
    lastFlowRate: mix(first.lastFlowRate, second.lastFlowRate, amount),
    lastCaptured: mix(first.lastCaptured, second.lastCaptured, amount),
    streamEnd: mixArray(first.streamEnd, second.streamEnd, amount),
    elapsedTime: mix(first.elapsedTime, second.elapsedTime, amount),
  };
}

function progressAt(now) {
  if (!fromState || !targetState || segmentDuration <= 0) {
    return 1;
  }
  return clamp((now - segmentStart) / segmentDuration, 0, 1);
}

function stateAt(now) {
  if (!fromState || !targetState) {
    return null;
  }
  return blendedState(fromState, targetState, progressAt(now));
}

function freezeAt(current, metadata) {
  return {
    ...metadata,
    q: current.q.slice(),
    fill: current.fill,
    spill: current.spill,
    lastFlow: current.lastFlow,
    lastFlowRate: current.lastFlowRate,
    lastCaptured: current.lastCaptured,
    streamEnd: current.streamEnd.slice(),
    elapsedTime: current.elapsedTime,
  };
}

function ingestSnapshot() {
  let raw;
  try {
    raw = typeof props.value === "string" ? JSON.parse(props.value) : props.value;
  } catch (error) {
    return;
  }
  if (!raw || raw.schema_version !== 1 || !raw.state || !raw.playback) {
    return;
  }

  const next = normalizedSnapshot(raw);
  if (
    !Array.isArray(next.q) || next.q.length !== 6 ||
    !next.q.every(Number.isFinite) || !Number.isFinite(next.intervalMs) ||
    next.intervalMs <= 0 || next.generation < lastGeneration
  ) {
    return;
  }
  const isNewRun = next.generation > lastGeneration;
  if (!isNewRun && next.revision <= lastRevision) {
    return;
  }

  const now = performance.now();
  if (isNewRun || !targetState) {
    fromState = next;
    targetState = next;
    segmentStart = now;
    segmentDuration = 0;
  } else {
    const amount = progressAt(now);
    const current = stateAt(now);
    const sameDecision = next.step === targetState.step;
    const speedChanged = next.speed !== targetState.speed;
    const wasPaused = targetState.paused;
    if (next.paused) {
      const frozen = freezeAt(current, next);
      fromState = frozen;
      targetState = frozen;
      segmentStart = now;
      segmentDuration = 0;
    } else if (sameDecision && !next.running) {
      // Manual finish has no new decision endpoint, so show the exact state.
      fromState = next;
      targetState = next;
      segmentStart = now;
      segmentDuration = 0;
    } else if (sameDecision && !speedChanged && !wasPaused) {
      // Motor/status changes apply at the next decision.  Accept their newer
      // metadata without restarting the animation already in progress.
      targetState = {
        ...targetState,
        generation: next.generation,
        revision: next.revision,
        running: next.running,
        paused: next.paused,
      };
    } else {
      fromState = current;
      targetState = next;
      segmentStart = now;
      const remaining = sameDecision && speedChanged && !wasPaused ? 1 - amount : 1;
      segmentDuration = prefersReducedMotion
        ? 0
        : Math.max(0, next.intervalMs * remaining);
    }
  }
  lastGeneration = next.generation;
  lastRevision = next.revision;
  drawFrame(stateAt(now));
  scheduleAnimation();
}

function rotate(point, angle) {
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  return [
    cosine * point[0] - sine * point[1],
    sine * point[0] + cosine * point[1],
  ];
}

function add(first, second) {
  return [first[0] + second[0], first[1] + second[1]];
}

function subtract(first, second) {
  return [first[0] - second[0], first[1] - second[1]];
}

function armPoints(base, q1, q2, lengths) {
  const elbow = add(base, [
    lengths[0] * Math.cos(q1),
    lengths[0] * Math.sin(q1),
  ]);
  const wrist = add(elbow, [
    lengths[1] * Math.cos(q1 + q2),
    lengths[1] * Math.sin(q1 + q2),
  ]);
  return [base.slice(), elbow, wrist];
}

function transformedPoints(center, angle, localPoints) {
  return localPoints.map((point) => add(center, rotate(point, angle)));
}

function canvasContext() {
  const canvas = element.querySelector(".coffee-canvas");
  if (!canvas) {
    return null;
  }
  if (canvas !== canvasRef) {
    canvasRef = canvas;
    if (resizeObserver) {
      resizeObserver.disconnect();
    }
    resizeObserver = new ResizeObserver(() => drawFrame(stateAt(performance.now())));
    resizeObserver.observe(canvas);
  }
  const width = Math.max(320, canvas.clientWidth || LOGICAL_WIDTH);
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(width * ratio);
  const pixelHeight = Math.round(width * LOGICAL_HEIGHT / LOGICAL_WIDTH * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const context = canvas.getContext("2d");
  context.setTransform(
    pixelWidth / LOGICAL_WIDTH,
    0,
    0,
    pixelHeight / LOGICAL_HEIGHT,
    0,
    0
  );
  return context;
}

function polygonPath(context, points, xy) {
  context.beginPath();
  points.forEach((point, index) => {
    const pixel = xy(point);
    if (index === 0) {
      context.moveTo(pixel[0], pixel[1]);
    } else {
      context.lineTo(pixel[0], pixel[1]);
    }
  });
  context.closePath();
}

function strokeWorldLine(context, points, xy, color, width) {
  context.beginPath();
  points.forEach((point, index) => {
    const pixel = xy(point);
    if (index === 0) {
      context.moveTo(pixel[0], pixel[1]);
    } else {
      context.lineTo(pixel[0], pixel[1]);
    }
  });
  context.strokeStyle = color;
  context.lineWidth = width;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.stroke();
}

function roundedRectangle(context, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + r, y);
  context.lineTo(x + width - r, y);
  context.quadraticCurveTo(x + width, y, x + width, y + r);
  context.lineTo(x + width, y + height - r);
  context.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  context.lineTo(x + r, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - r);
  context.lineTo(x, y + r);
  context.quadraticCurveTo(x, y, x + r, y);
  context.closePath();
}

function drawFrame(state) {
  if (!state) {
    return;
  }
  const context = canvasContext();
  if (!context) {
    return;
  }

  const width = LOGICAL_WIDTH;
  const height = LOGICAL_HEIGHT;
  const header = 76;
  const footer = 50;
  const bottom = height - footer;
  const geometry = state.geometry;
  const xBounds = geometry.world_bounds_m.x;
  const yBounds = geometry.world_bounds_m.y;
  const scale = Math.min(
    (width - 80) / (xBounds[1] - xBounds[0]),
    (height - header - footer) / (yBounds[1] - yBounds[0])
  );
  const centerX = width / 2;
  const xy = (point) => [centerX + point[0] * scale, bottom - point[1] * scale];

  const navy = "#224a67";
  const teal = "#2b7a78";
  const dark = "#1d2b3a";
  const muted = "#52656f";
  const coffee = "#70442b";
  const amber = "#bd8b29";
  const red = "#b4523b";

  context.clearRect(0, 0, width, height);
  context.fillStyle = "#f5f7f8";
  context.fillRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, header);

  context.textBaseline = "alphabetic";
  context.fillStyle = teal;
  context.font = "700 14px ui-sans-serif, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("KAIST OR GYM", 32, 30);
  context.fillStyle = dark;
  context.font = "700 25px ui-sans-serif, system-ui, sans-serif";
  context.fillText("Two-arm coffee pouring", 32, 61);
  context.fillStyle = navy;
  context.font = "700 15px ui-sans-serif, system-ui, sans-serif";
  context.textAlign = "right";
  context.fillText("fixed links · six revolute joints", width - 32, 42);

  context.strokeStyle = "#dfe5e7";
  context.lineWidth = 1;
  [-0.8, -0.4, 0, 0.4, 0.8].forEach((value) => {
    const pixel = xy([value, 0]);
    context.beginPath();
    context.moveTo(pixel[0], header);
    context.lineTo(pixel[0], bottom);
    context.stroke();
  });
  [0.2, 0.4, 0.6, 0.8, 1.0].forEach((value) => {
    const pixel = xy([0, value]);
    context.beginPath();
    context.moveTo(centerX - scale, pixel[1]);
    context.lineTo(centerX + scale, pixel[1]);
    context.stroke();
  });
  const tablePixel = xy([0, geometry.table_y_m])[1];
  context.fillStyle = "#dad5ca";
  context.fillRect(0, tablePixel, width, bottom - tablePixel);
  context.strokeStyle = "#8e9797";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(0, tablePixel);
  context.lineTo(width, tablePixel);
  context.stroke();

  const cupArm = geometry.arms.cup;
  const potArm = geometry.arms.pot;
  const cupJoints = armPoints(cupArm.base_m, state.q[0], state.q[1], cupArm.link_lengths_m);
  const potJoints = armPoints(potArm.base_m, state.q[3], state.q[4], potArm.link_lengths_m);

  function drawArm(points, color, label) {
    strokeWorldLine(context, points, xy, "#b5c1c5", 20);
    strokeWorldLine(context, points, xy, color, 12);
    points.forEach((point) => {
      const pixel = xy(point);
      context.beginPath();
      context.arc(pixel[0], pixel[1], 10, 0, 2 * Math.PI);
      context.fillStyle = "#fafcfc";
      context.fill();
      context.strokeStyle = color;
      context.lineWidth = 4;
      context.stroke();
    });
    const base = xy(points[0]);
    context.fillStyle = muted;
    context.font = "700 12px ui-sans-serif, system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText(label, base[0], base[1] + 27);
  }

  drawArm(cupJoints, teal, "cup arm");
  drawArm(potJoints, navy, "pot arm");

  const cupTool = geometry.tools.cup;
  const potTool = geometry.tools.pot;
  const cupAngle = state.q[0] + state.q[1] + state.q[2];
  const potAngle = state.q[3] + state.q[4] + state.q[5];
  const cupCenter = subtract(cupJoints[2], rotate(cupTool.grip_offset_m, cupAngle));
  const potCenter = subtract(potJoints[2], rotate(potTool.grip_offset_m, potAngle));
  const cupMouth = add(cupCenter, rotate(cupTool.landmark_offset_m, cupAngle));
  const potSpout = add(potCenter, rotate(potTool.landmark_offset_m, potAngle));
  const cupWidth = cupTool.size_m[0];
  const cupHeight = cupTool.size_m[1];
  const potWidth = potTool.size_m[0];
  const potHeight = potTool.size_m[1];

  const cupOuter = transformedPoints(cupCenter, cupAngle, [
    [-0.50 * cupWidth, 0.50 * cupHeight],
    [0.50 * cupWidth, 0.50 * cupHeight],
    [0.38 * cupWidth, -0.50 * cupHeight],
    [-0.38 * cupWidth, -0.50 * cupHeight],
  ]);
  polygonPath(context, cupOuter, xy);
  context.fillStyle = "#fafcfc";
  context.fill();
  context.save();
  polygonPath(context, cupOuter, xy);
  context.clip();
  const fillRatio = clamp(state.fill / 1.02, 0, 1);
  const surfaceY = cupCenter[1] - 0.5 * cupHeight + fillRatio * cupHeight;
  const surfacePixel = xy([0, surfaceY])[1];
  context.fillStyle = coffee;
  context.fillRect(0, surfacePixel, width, height - surfacePixel);
  context.restore();
  polygonPath(context, cupOuter, xy);
  context.strokeStyle = muted;
  context.lineWidth = 3;
  context.stroke();

  const targetRatio = clamp(state.targetFill / 1.02, 0, 1);
  const targetY = cupCenter[1] - 0.5 * cupHeight + targetRatio * cupHeight;
  strokeWorldLine(context, [
    [cupCenter[0] - 0.32 * cupWidth, targetY],
    [cupCenter[0] + 0.32 * cupWidth, targetY],
  ], xy, navy, 2);

  const cupHandle = [];
  for (let index = 0; index < 24; index += 1) {
    const angle = -Math.PI / 2 + index * Math.PI / 23;
    cupHandle.push([
      0.45 * cupWidth + 0.34 * cupWidth * Math.cos(angle),
      0.25 * cupHeight * Math.sin(angle),
    ]);
  }
  strokeWorldLine(
    context,
    transformedPoints(cupCenter, cupAngle, cupHandle),
    xy,
    muted,
    4
  );

  const potOuter = transformedPoints(potCenter, potAngle, [
    [-0.47 * potWidth, 0.48 * potHeight],
    [0.47 * potWidth, 0.48 * potHeight],
    [0.43 * potWidth, -0.48 * potHeight],
    [-0.43 * potWidth, -0.48 * potHeight],
  ]);
  polygonPath(context, potOuter, xy);
  context.fillStyle = "#798487";
  context.fill();
  context.strokeStyle = "#394f58";
  context.lineWidth = 2;
  context.stroke();

  const spoutShape = transformedPoints(potCenter, potAngle, [
    [-0.43 * potWidth, 0.34 * potHeight],
    [-0.78 * potWidth, 0.22 * potHeight],
    [-0.43 * potWidth, 0.06 * potHeight],
  ]);
  polygonPath(context, spoutShape, xy);
  context.fillStyle = "#56666a";
  context.fill();
  context.strokeStyle = "#394f58";
  context.stroke();

  const coffeeWindow = transformedPoints(potCenter, potAngle, [
    [-0.30 * potWidth, 0.10 * potHeight],
    [0.28 * potWidth, 0.10 * potHeight],
    [0.28 * potWidth, -0.26 * potHeight],
    [-0.30 * potWidth, -0.26 * potHeight],
  ]);
  polygonPath(context, coffeeWindow, xy);
  context.fillStyle = coffee;
  context.fill();

  const potHandle = [];
  for (let index = 0; index < 24; index += 1) {
    const angle = -Math.PI / 2 + index * Math.PI / 23;
    potHandle.push([
      0.43 * potWidth + 0.38 * potWidth * Math.cos(angle),
      0.30 * potHeight * Math.sin(angle),
    ]);
  }
  strokeWorldLine(
    context,
    transformedPoints(potCenter, potAngle, potHandle),
    xy,
    "#394f58",
    7
  );

  if (state.lastFlow > 1e-5) {
    const streamColor = state.lastCaptured > 0.5 * state.lastFlow ? coffee : red;
    strokeWorldLine(context, [potSpout, state.streamEnd], xy, streamColor, 5);
  }
  if (state.spill > 0.002) {
    const puddle = xy([0, geometry.table_y_m]);
    const puddleRadius = clamp(26 + 140 * state.spill, 26, 90);
    context.beginPath();
    context.ellipse(puddle[0], puddle[1], puddleRadius, 7, 0, 0, 2 * Math.PI);
    context.fillStyle = "#94644a";
    context.fill();
  }

  roundedRectangle(context, 25, height - 42, width - 50, 32, 8);
  context.fillStyle = "#ffffff";
  context.fill();
  context.font = "700 14px ui-sans-serif, system-ui, sans-serif";
  context.textBaseline = "middle";
  context.textAlign = "left";
  context.fillStyle = dark;
  context.fillText(
    "fill " + Math.round(state.fill * 1000) + "/" +
      Math.round(state.targetFill * 1000) + " mL",
    40,
    height - 26
  );
  context.textAlign = "center";
  context.fillStyle = state.spill > 0.02 ? red : muted;
  context.fillText("spill " + Math.round(state.spill * 1000) + " mL", width / 2, height - 26);
  context.textAlign = "right";
  context.fillStyle = amber;
  const timeText = state.horizon === null
    ? "decision " + state.step + " · " + state.elapsedTime.toFixed(2) + " s"
    : "decision " + state.step + "/" + state.horizon + " · " +
      Math.max(0, (state.horizon - state.step) * state.dt).toFixed(1) + " s left";
  context.fillText(timeText, width - 40, height - 26);

  // The landmark is derived from the interpolated joint state, never lerped.
  void cupMouth;
}

function scheduleAnimation() {
  if (!animationHandle) {
    animationHandle = window.requestAnimationFrame(animationLoop);
  }
}

function animationLoop(now) {
  animationHandle = 0;
  if (!element.isConnected) {
    if (resizeObserver) {
      resizeObserver.disconnect();
    }
    return;
  }
  drawFrame(stateAt(now));
  if (progressAt(now) < 1) {
    scheduleAnimation();
  }
}

watch("value", ingestSnapshot);
ingestSnapshot();
scheduleAnimation();
"""
