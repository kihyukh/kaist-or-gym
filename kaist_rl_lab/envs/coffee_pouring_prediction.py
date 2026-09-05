"""Bounded visual prediction; Python remains authoritative for physics and data."""

PREDICTION_JAVASCRIPT = r"""
// Predict only the rigid pose while a control request is in flight. All liquid,
// rewards and recorded transitions still come from Python. Conservative contact
// checks stop the preview at obstacles; authoritative snapshots reconcile it.
function previewShapes(q, g, name) {
  const cup = name === "cup", offset = cup ? 0 : 3;
  const arm = g.arms[name], tool = g.tools[name];
  const points = armPoints(arm.base_m, q[offset], q[offset + 1], arm.link_lengths_m);
  const angle = q[offset] + q[offset + 1] + q[offset + 2];
  const center = subtract(points[2], rotate(tool.grip_offset_m, angle));
  const [w, h] = tool.size_m;
  const polygons = cup
    ? [[[-.5, .5], [.5, .5], [.38, -.5], [-.38, -.5]]]
    : [[[-.47, .48], [.47, .48], [.43, -.48], [-.43, -.48]],
       [[-.43, .34], [-.78, .22], [-.43, .06]]];
  const bodies = polygons.map(p => transformedPoints(center, angle, p.map(([x,y]) => [x*w,y*h])));
  const c = g.collision;
  const radius = cup ? c.cup_handle_radius_m : c.pot_handle_radius_m;
  const handle = Array.from({length: 25}, (_, i) => {
    const a = -Math.PI / 2 + Math.PI * i / 24;
    return add(center, rotate([
      ((cup ? .45 : .43) + (cup ? .34 : .38) * Math.cos(a))*w,
      (cup ? .25 : .30)*Math.sin(a)*h,
    ], angle));
  });
  const segments = points.slice(1).map((p,i) => [points[i], p, c.link_radius_m]);
  handle.slice(1).forEach((p,i) => segments.push([handle[i], p, radius]));
  return {bodies, segments};
}

function pointSegmentDistance(p, a, b) {
  const dx=b[0]-a[0], dy=b[1]-a[1], length=dx*dx+dy*dy;
  const t=length ? clamp(((p[0]-a[0])*dx+(p[1]-a[1])*dy)/length,0,1) : 0;
  return Math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy);
}
function cross(a,b,c) { return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]); }
function segmentDistance(a,b,c,d) {
  const abC=cross(a,b,c), abD=cross(a,b,d), cdA=cross(c,d,a), cdB=cross(c,d,b);
  if (abC*abD<0 && cdA*cdB<0) return 0;
  return Math.min(pointSegmentDistance(a,c,d),pointSegmentDistance(b,c,d),
    pointSegmentDistance(c,a,b),pointSegmentDistance(d,a,b));
}
function polygonEdges(p) { return p.map((a,i) => [a,p[(i+1)%p.length]]); }
function pointInPolygon(point,p) {
  const signs=polygonEdges(p).map(([a,b]) => cross(a,b,point));
  return signs.every(x => x>=0) || signs.every(x => x<=0);
}
function boundsOverlap(a,b,margin) {
  return [0,1].every(i => Math.max(...a.map(p=>p[i]))+margin>=Math.min(...b.map(p=>p[i])) &&
    Math.max(...b.map(p=>p[i]))+margin>=Math.min(...a.map(p=>p[i])));
}
function capsulePolygon(a,b,r,p) {
  if (!boundsOverlap([a,b],p,r)) return false;
  return pointInPolygon(a,p) || pointInPolygon(b,p) ||
    polygonEdges(p).some(([c,d]) => segmentDistance(a,b,c,d)<r);
}
function previewCollision(q,g) {
  const cup=previewShapes(q,g,"cup"), pot=previewShapes(q,g,"pot"), c=g.collision;
  for (const shape of [cup,pot]) {
    if (shape.segments.some(([a,b,r]) => Math.min(a[1],b[1])-r<g.table_y_m)) return true;
    if (shape.bodies.some(p => Math.min(...p.map(v=>v[1]))-c.table_margin_m<g.table_y_m)) return true;
  }
  for (const [a,b,r] of cup.segments) {
    for (const [d,e,s] of pot.segments) {
      if (boundsOverlap([a,b],[d,e],r+s) && segmentDistance(a,b,d,e)<r+s) return true;
    }
    if (pot.bodies.some(p => capsulePolygon(a,b,r+c.body_margin_m,p))) return true;
  }
  for (const [a,b,r] of pot.segments) {
    if (cup.bodies.some(p => capsulePolygon(a,b,r+c.body_margin_m,p))) return true;
  }
  for (const a of cup.bodies) for (const b of pot.bodies) {
    if (!boundsOverlap(a,b,c.body_margin_m)) continue;
    if (a.some(p=>pointInPolygon(p,b)) || b.some(p=>pointInPolygon(p,a))) return true;
    if (polygonEdges(a).some(([p,q]) => polygonEdges(b).some(([r,s]) =>
      segmentDistance(p,q,r,s)<c.body_margin_m))) return true;
  }
  return false;
}
function predictPose(q, motors, seconds, g) {
  if (!g.collision || seconds<=0) return q.slice();
  const {low,high}=g.joint_limits_rad;
  const speeds=g.max_joint_speeds_rad_s;
  let current=q.slice();
  // Small increments prevent preview tunnelling and keep rigid links intact.
  const steps=Math.ceil(seconds/(1/64)), dt=seconds/steps;
  for (let n=0;n<steps;n++) {
    const next=current.map((v,i)=>clamp(v+motors[i]*speeds[i]*dt,low[i],high[i]));
    if (next.every((v,i)=>v===current[i])) break;
    if (previewCollision(next,g)) break;
    current=next;
  }
  return current;
}
"""
