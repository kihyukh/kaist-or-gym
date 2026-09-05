"""Draw authoritative Python snapshots with rigid forward kinematics."""

CANVAS_HTML = """
<div class="coffee-stage">
  <div class="coffee-canvas-wrap">
    <canvas class="coffee-canvas" role="img"
      aria-label="Two fixed-link robot arms pouring coffee"></canvas>
  </div>
  <div class="coffee-control-dock" role="group" aria-label="Robot joint controls">
    <section class="coffee-arm-controls cup-joint" aria-label="Cup arm">
      <h3 class="coffee-arm-label">Cup arm <span>Left robot</span></h3>
      <div class="coffee-arm-joints">
        <div class="coffee-joint-control cup-joint" data-joint-index="0">
          <div class="coffee-joint-card">
            <span class="coffee-joint-label">Shoulder</span>
            <div class="coffee-joint-buttons" role="group" aria-label="Cup shoulder controls">
              <button type="button" disabled class="coffee-joint-button" data-direction="1"
                aria-label="Rotate cup shoulder counter-clockwise" title="Rotate counter-clockwise">↺</button>
              <button type="button" disabled class="coffee-joint-button" data-direction="0"
                aria-label="Hold cup shoulder" title="Hold this joint">■</button>
              <button type="button" disabled class="coffee-joint-button" data-direction="-1"
                aria-label="Rotate cup shoulder clockwise" title="Rotate clockwise">↻</button>
            </div>
          </div>
        </div>
        <div class="coffee-joint-control cup-joint" data-joint-index="1">
          <div class="coffee-joint-card">
            <span class="coffee-joint-label">Elbow</span>
            <div class="coffee-joint-buttons" role="group" aria-label="Cup elbow controls">
              <button type="button" disabled class="coffee-joint-button" data-direction="1"
                aria-label="Rotate cup elbow counter-clockwise" title="Rotate counter-clockwise">↺</button>
              <button type="button" disabled class="coffee-joint-button" data-direction="0"
                aria-label="Hold cup elbow" title="Hold this joint">■</button>
              <button type="button" disabled class="coffee-joint-button" data-direction="-1"
                aria-label="Rotate cup elbow clockwise" title="Rotate clockwise">↻</button>
            </div>
          </div>
        </div>
        <div class="coffee-joint-control cup-joint" data-joint-index="2">
          <div class="coffee-joint-card">
            <span class="coffee-joint-label">Wrist</span>
            <div class="coffee-joint-buttons" role="group" aria-label="Cup wrist controls">
              <button type="button" disabled class="coffee-joint-button" data-direction="1"
                aria-label="Rotate cup wrist counter-clockwise" title="Rotate counter-clockwise">↺</button>
              <button type="button" disabled class="coffee-joint-button" data-direction="0"
                aria-label="Hold cup wrist" title="Hold this joint">■</button>
              <button type="button" disabled class="coffee-joint-button" data-direction="-1"
                aria-label="Rotate cup wrist clockwise" title="Rotate clockwise">↻</button>
            </div>
          </div>
        </div>
      </div>
    </section>
    <section class="coffee-arm-controls pot-joint" aria-label="Pot arm">
      <h3 class="coffee-arm-label">Pot arm <span>Right robot</span></h3>
      <div class="coffee-arm-joints">
        <div class="coffee-joint-control pot-joint" data-joint-index="3">
          <div class="coffee-joint-card">
            <span class="coffee-joint-label">Shoulder</span>
            <div class="coffee-joint-buttons" role="group" aria-label="Coffee pot shoulder controls">
              <button type="button" disabled class="coffee-joint-button" data-direction="1"
                aria-label="Rotate coffee pot shoulder counter-clockwise" title="Rotate counter-clockwise">↺</button>
              <button type="button" disabled class="coffee-joint-button" data-direction="0"
                aria-label="Hold coffee pot shoulder" title="Hold this joint">■</button>
              <button type="button" disabled class="coffee-joint-button" data-direction="-1"
                aria-label="Rotate coffee pot shoulder clockwise" title="Rotate clockwise">↻</button>
            </div>
          </div>
        </div>
        <div class="coffee-joint-control pot-joint" data-joint-index="4">
          <div class="coffee-joint-card">
            <span class="coffee-joint-label">Elbow</span>
            <div class="coffee-joint-buttons" role="group" aria-label="Coffee pot elbow controls">
              <button type="button" disabled class="coffee-joint-button" data-direction="1"
                aria-label="Rotate coffee pot elbow counter-clockwise" title="Rotate counter-clockwise">↺</button>
              <button type="button" disabled class="coffee-joint-button" data-direction="0"
                aria-label="Hold coffee pot elbow" title="Hold this joint">■</button>
              <button type="button" disabled class="coffee-joint-button" data-direction="-1"
                aria-label="Rotate coffee pot elbow clockwise" title="Rotate clockwise">↻</button>
            </div>
          </div>
        </div>
        <div class="coffee-joint-control pot-joint" data-joint-index="5">
          <div class="coffee-joint-card">
            <span class="coffee-joint-label">Wrist</span>
            <div class="coffee-joint-buttons" role="group" aria-label="Coffee pot wrist controls">
              <button type="button" disabled class="coffee-joint-button" data-direction="1"
                aria-label="Rotate coffee pot wrist counter-clockwise" title="Rotate counter-clockwise">↺</button>
              <button type="button" disabled class="coffee-joint-button" data-direction="0"
                aria-label="Hold coffee pot wrist" title="Hold this joint">■</button>
              <button type="button" disabled class="coffee-joint-button" data-direction="-1"
                aria-label="Rotate coffee pot wrist clockwise" title="Rotate clockwise">↻</button>
            </div>
          </div>
        </div>
      </div>
    </section>
  </div>
  <div class="coffee-render-note">
    ↺ Counterclockwise · ■ Hold · ↻ Clockwise. Each command stays active until changed.
  </div>
</div>
"""

CANVAS_CSS = """
.coffee-stage {
  width: 100%;
  container-type: inline-size;
  overflow: hidden;
  border: 1px solid #d8e0e3;
  border-radius: 9px;
  background: #f5f7f8;
  box-shadow: 0 1px 2px rgba(29, 43, 58, 0.08);
}
.coffee-canvas-wrap {
  width: 100%;
}
.coffee-canvas {
  display: block;
  width: min(100%, 960px, max(560px, calc((100svh - 330px) * 1.714286)));
  margin-inline: auto;
  aspect-ratio: 12 / 7;
  background: #f5f7f8;
}
.coffee-control-dock {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
  padding: 14px 16px;
  border-top: 1px solid #d8e0e3;
  background: #ffffff;
}
.coffee-arm-controls {
  min-width: 0;
  padding-top: 9px;
  border-top: 3px solid #2b7a78;
}
.coffee-arm-controls.pot-joint {
  border-color: #224a67;
}
.coffee-arm-label {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 8px;
  margin: 0 0 9px;
  color: #2b7a78;
  font: 700 14px/1.2 ui-sans-serif, system-ui, sans-serif;
}
.pot-joint .coffee-arm-label {
  color: #224a67;
}
.coffee-arm-label span {
  color: #52656f;
  font-size: 11px;
  font-weight: 500;
}
.coffee-arm-joints {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}
.coffee-joint-control {
  min-width: 0;
}
.coffee-joint-card {
  padding: 8px;
  border: 1px solid #d8e0e3;
  border-radius: 8px;
  background: #f5f7f8;
}
.coffee-joint-label {
  display: block;
  margin-bottom: 7px;
  color: #344b58;
  font: 600 12px/1.1 ui-sans-serif, system-ui, sans-serif;
  letter-spacing: 0.025em;
  text-align: center;
  white-space: nowrap;
}
.coffee-joint-buttons {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  justify-content: center;
  gap: 4px;
}
.coffee-joint-button {
  width: 100%;
  min-width: 0;
  height: 36px;
  margin: 0;
  padding: 0;
  border: 1px solid #aab8be;
  border-radius: 5px;
  background: #ffffff;
  color: #344b58;
  cursor: pointer;
  font: 800 17px/1 ui-sans-serif, system-ui, sans-serif;
  touch-action: manipulation;
  transition: background 90ms ease, border-color 90ms ease, color 90ms ease,
    box-shadow 90ms ease, transform 70ms ease;
}
.coffee-joint-button:hover:not(:disabled) {
  border-color: #52656f;
  background: #f0f4f5;
}
.coffee-joint-button:active:not(:disabled) {
  transform: translateY(1px);
}
.coffee-joint-button:focus-visible {
  outline: 3px solid rgba(189, 139, 41, 0.45);
  outline-offset: 1px;
}
.coffee-joint-button:disabled {
  cursor: not-allowed;
  opacity: 0.48;
}
.cup-joint .coffee-joint-button.is-active {
  border-color: #2b7a78;
  background: #2b7a78;
  color: #ffffff;
  box-shadow: 0 0 0 2px rgba(43, 122, 120, 0.18);
}
.pot-joint .coffee-joint-button.is-active {
  border-color: #224a67;
  background: #224a67;
  color: #ffffff;
  box-shadow: 0 0 0 2px rgba(34, 74, 103, 0.18);
}
.coffee-render-note {
  display: flex;
  box-sizing: border-box;
  align-items: center;
  gap: 7px;
  padding: 7px 12px 8px;
  border-top: 1px solid #e1e7e9;
  background: #ffffff;
  color: #52656f;
  font: 600 12px/1.2 ui-sans-serif, system-ui, sans-serif;
  letter-spacing: 0.01em;
}
@container (max-width: 760px) {
  .coffee-control-dock {
    grid-template-columns: minmax(0, 1fr);
    gap: 14px;
    padding: 12px;
  }
  .coffee-joint-button {
    height: 44px;
  }
}
@container (max-width: 420px) {
  .coffee-joint-card {
    padding: 7px 4px;
  }
  .coffee-arm-joints {
    gap: 5px;
  }
}
"""

CANVAS_JAVASCRIPT = r"""
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
    cupCapacity: liquid.cup_capacity_l,
    stableCupCapacity: liquid.stable_cup_capacity_l,
    sourceInitial: liquid.source_initial_l,
    sourceCapacity: liquid.source_capacity_l,
    sourceRemaining: liquid.source_remaining_l,
    lastFlow: liquid.last_flow_l,
    lastFlowRate: liquid.last_flow_rate_l_s,
    lastCaptured: liquid.last_captured_l,
    lastCaptureFraction: liquid.last_capture_fraction,
    lastPourIntensity: liquid.last_pour_intensity,
    lastExitSpeed: liquid.last_exit_speed_m_s,
    lastJetRadius: liquid.last_jet_radius_m,
    cupSurfaceY: liquid.cup_surface_y_m,
    targetSurfaceY: liquid.target_surface_y_m,
    potSurfaceY: liquid.pot_surface_y_m,
    streamEnd: liquid.stream_end_m.slice(),
    streamPath: liquid.stream_path_m.map((point) => point.slice()),
    spillPath: liquid.spill_path_m.map((point) => point.slice()),
    directSpill: liquid.direct_spill_l,
    directSpillRate: liquid.direct_spill_rate_l_s,
    directSpillPath: liquid.direct_spill_path_m.map((point) => point.slice()),
    cupRunoff: liquid.cup_runoff_l,
    cupRunoffRate: liquid.cup_runoff_rate_l_s,
    cupRunoffPath: liquid.cup_runoff_path_m.map((point) => point.slice()),
    spillImpactX: liquid.spill_impact_x_m,
    step: state.step,
    elapsedTime: state.elapsed_time_s,
    dt: state.dt_s,
    horizon: state.horizon_steps,
    terminationReason: state.termination_reason,
    generation: playback.generation,
    revision: playback.revision,
    inputSequence: playback.input_sequence || 0,
    speed: playback.speed,
    paused: playback.paused,
    running: playback.running,
    motors: playback.motors.slice(),
    intervalMs: playback.decision_interval_wall_ms,
  };
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

function horizontalSpan(points, worldY) {
  const intersections = [];
  points.forEach((first, index) => {
    const second = points[(index + 1) % points.length];
    const deltaY = second[1] - first[1];
    if (Math.abs(deltaY) < 1e-12) {
      if (Math.abs(first[1] - worldY) < 1e-12) {
        intersections.push(first[0], second[0]);
      }
      return;
    }
    const fraction = (worldY - first[1]) / deltaY;
    if (fraction >= -1e-12 && fraction <= 1 + 1e-12) {
      intersections.push(first[0] + fraction * (second[0] - first[0]));
    }
  });
  if (intersections.length < 2) {
    const centre = points.reduce((sum, point) => sum + point[0], 0) / points.length;
    return [centre, centre];
  }
  return [Math.min(...intersections), Math.max(...intersections)];
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
    resizeObserver = new ResizeObserver(() => drawFrame(displayState));
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

function updateJointControls(state) {
  element.querySelectorAll(".coffee-joint-control").forEach((control) => {
    const index = Number(control.dataset.jointIndex);
    control.querySelectorAll(".coffee-joint-button").forEach((button) => {
      const selected = Number(button.dataset.direction) === state.motors[index];
      button.classList.toggle("is-active", selected);
      button.setAttribute("aria-pressed", String(selected));
      button.disabled = !state.running;
    });
  });
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
  updateJointControls(state);

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
  const surfacePixel = xy([0, state.cupSurfaceY])[1];
  context.fillStyle = coffee;
  context.fillRect(0, surfacePixel, width, height - surfacePixel);
  context.restore();
  polygonPath(context, cupOuter, xy);
  context.strokeStyle = muted;
  context.lineWidth = 3;
  context.stroke();

  const targetY = state.targetSurfaceY;
  const targetSpan = horizontalSpan(cupOuter, targetY);
  strokeWorldLine(context, [
    [targetSpan[0], targetY],
    [targetSpan[1], targetY],
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
    [-0.43 * potWidth, 0.45 * potHeight],
    [0.43 * potWidth, 0.45 * potHeight],
    [0.43 * potWidth, -0.45 * potHeight],
    [-0.43 * potWidth, -0.45 * potHeight],
  ]);
  context.save();
  polygonPath(context, coffeeWindow, xy);
  context.clip();
  context.fillStyle = coffee;
  const sourceSurfacePixel = xy([0, state.potSurfaceY])[1];
  context.fillRect(0, sourceSurfacePixel, width, height - sourceSurfacePixel);
  context.restore();
  polygonPath(context, coffeeWindow, xy);
  context.strokeStyle = "#394f58";
  context.lineWidth = 1.5;
  context.stroke();

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

  const showTransientFlow = state.running || state.transitioning;
  const streamOffset = subtract(potSpout, state.streamPath[0]);
  const streamPath = state.streamPath.map((point) => add(point, streamOffset));
  if (showTransientFlow && state.lastFlowRate > 1e-5) {
    const streamWidth = clamp(2 * state.lastJetRadius * scale, 1.25, 6);
    strokeWorldLine(context, streamPath, xy, coffee, streamWidth);
  }
  if (showTransientFlow && !state.paused && state.directSpill > 1e-8) {
    const directRadius = clamp(
      Math.sqrt((state.directSpillRate / 1000) / (Math.PI * 0.35)),
      0,
      0.0055
    );
    const directWidth = clamp(2 * directRadius * scale, 1.25, 6);
    strokeWorldLine(context, state.directSpillPath, xy, coffee, directWidth);
  }
  if (showTransientFlow && state.cupRunoff > 1e-8) {
    const pathStart = state.cupRunoffPath[0];
    const nearestRim = Math.hypot(
      cupOuter[0][0] - pathStart[0],
      cupOuter[0][1] - pathStart[1]
    ) <= Math.hypot(
      cupOuter[1][0] - pathStart[0],
      cupOuter[1][1] - pathStart[1]
    ) ? cupOuter[0] : cupOuter[1];
    const runoffOffset = subtract(nearestRim, pathStart);
    const cupRunoffPath = state.cupRunoffPath.map(
      (point) => add(point, runoffOffset)
    );
    const runoffRadius = clamp(
      Math.sqrt((state.cupRunoffRate / 1000) / (Math.PI * 0.35)),
      0,
      0.0055
    );
    const runoffWidth = clamp(2 * runoffRadius * scale, 1.25, 6);
    strokeWorldLine(context, cupRunoffPath, xy, coffee, runoffWidth);
  }
  if (state.spill > 1e-8) {
    const puddle = xy([state.spillImpactX, geometry.table_y_m]);
    const puddleWorldRadius = clamp(
      0.13 * Math.sqrt(state.spill / 0.10),
      0.005,
      0.26
    );
    const puddleRadius = Math.max(2, puddleWorldRadius * scale);
    const puddleHeight = Math.max(2, 0.16 * puddleRadius);
    context.beginPath();
    context.ellipse(
      puddle[0],
      puddle[1],
      puddleRadius,
      puddleHeight,
      0,
      0,
      2 * Math.PI
    );
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
    "cup " + Math.round(state.fill * 1000) + "/" +
      Math.round(state.targetFill * 1000) + " mL · pot " +
      Math.round(state.sourceRemaining * 1000) + " mL",
    40,
    height - 26
  );
  context.textAlign = "center";
  context.fillStyle = state.spill > 0.02 ? red : muted;
  const displayFlowRate = showTransientFlow ? state.lastFlowRate : 0;
  context.fillText(
    "spill " + Math.round(state.spill * 1000) + " mL · pour " +
      Math.round(displayFlowRate * 1000) + " mL/s",
    width / 2,
    height - 26
  );
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

"""
