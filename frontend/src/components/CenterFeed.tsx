import { useEffect, useRef, useState } from "react";
import { Pause, Play, Crosshair, Layers } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ContextAuthorDialog } from "@/components/ContextAuthorDialog";
import { useStore } from "@/store/useStore";
import { fmtClock } from "@/lib/utils";
import * as api from "@/lib/api";
import type { TrackedObject } from "@/lib/types";

/** How far past the last detection we're willing to extrapolate a box, in
 *  seconds. Beyond this the box freezes where it is: a frozen box honestly
 *  reads as "stale", whereas one that keeps coasting is an active lie about
 *  where the object is. Roughly 1.5x the expected detection period. */
const MAX_EXTRAPOLATE_S = 0.35;

/** Draws the live UAV frame + SAM-3 style overlays onto a canvas.
 *
 *  `trackAgeS` is how long ago the current track set arrived. Detection runs
 *  at a few Hz while this renders at ~60Hz, so box centres are extrapolated
 *  from each track's velocity to keep motion smooth between detections. Only
 *  the centre is extrapolated — never size, score or label, which have no
 *  velocity signal and would just jitter. The server is never asked to invent
 *  detections; this is purely presentational.
 */
function render(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  tracks: TrackedObject[],
  t: number,
  frameImg: HTMLImageElement | null,
  trackAgeS: number,
) {
  if (frameImg && frameImg.complete && frameImg.naturalWidth > 0) {
    // real decoded video frame (backend/app/pipeline.py's "frame" messages)
    ctx.drawImage(frameImg, 0, 0, w, h);
  } else {
    // no stream connected yet — procedural placeholder, not real video
    const g = ctx.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, "#0a0f16");
    g.addColorStop(1, "#0d1420");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);

    ctx.globalAlpha = 0.5;
    for (let i = 0; i < 6; i++) {
      const bx = ((i * 137 + t * 6) % (w + 200)) - 100;
      const by = (i * 97) % h;
      const rad = ctx.createRadialGradient(bx, by, 0, bx, by, 120);
      rad.addColorStop(0, "rgba(60,90,120,0.25)");
      rad.addColorStop(1, "transparent");
      ctx.fillStyle = rad;
      ctx.beginPath();
      ctx.arc(bx, by, 120, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  // faint grid
  ctx.strokeStyle = "rgba(120,150,180,0.06)";
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // tracked objects: trail, mask glow, box, label
  const age = Math.min(trackAgeS, MAX_EXTRAPOLATE_S);
  // Fade boxes once we're past roughly one detection period, so an operator
  // can see at a glance which boxes are measured and which are predicted.
  const stale = trackAgeS > MAX_EXTRAPOLATE_S;
  for (const o of tracks) {
    // vx/vy are normalized units per second (see backend/app/tracker.py)
    const x = (o.x + o.vx * age) * w;
    const y = (o.y + o.vy * age) * h;
    const bw = o.w * w;
    const bh = o.h * h;
    ctx.globalAlpha = stale ? 0.55 : 1;

    if (o.trail.length > 1) {
      ctx.strokeStyle = o.color + "66";
      ctx.lineWidth = 2;
      ctx.beginPath();
      o.trail.forEach((p, i) => {
        const px = p.x * w;
        const py = p.y * h;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
    }

    // mask glow
    ctx.fillStyle = o.color + "22";
    ctx.strokeStyle = o.color;
    ctx.lineWidth = 1.5;
    roundRect(ctx, x - bw / 2, y - bh / 2, bw, bh, 4);
    ctx.fill();
    ctx.stroke();

    // corner ticks
    const cl = 7;
    ctx.strokeStyle = o.color;
    ctx.lineWidth = 2;
    const x0 = x - bw / 2;
    const y0 = y - bh / 2;
    corners(ctx, x0, y0, bw, bh, cl);

    // label chip
    const label = `${o.concept} #${o.id}`;
    ctx.font = "600 11px ui-monospace, monospace";
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = o.color;
    ctx.fillRect(x0, y0 - 15, tw + 10, 14);
    ctx.fillStyle = "#04121f";
    ctx.fillText(label, x0 + 5, y0 - 4);
    ctx.fillStyle = o.color;
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText(`${Math.round(o.score * 100)}%`, x0 + bw - 22, y0 + bh + 12);
  }
  ctx.globalAlpha = 1;

  // center reticle
  ctx.strokeStyle = "rgba(255,255,255,0.35)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(w / 2 - 16, h / 2);
  ctx.lineTo(w / 2 + 16, h / 2);
  ctx.moveTo(w / 2, h / 2 - 16);
  ctx.lineTo(w / 2, h / 2 + 16);
  ctx.stroke();
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function corners(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, l: number) {
  const seg = (ax: number, ay: number, bx: number, by: number, cx: number, cy: number) => {
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.lineTo(cx, cy);
    ctx.stroke();
  };
  seg(x, y + l, x, y, x + l, y);
  seg(x + w - l, y, x + w, y, x + w, y + l);
  seg(x, y + h - l, x, y + h, x + l, y + h);
  seg(x + w - l, y + h, x + w, y + h, x + w, y + h - l);
}

export function CenterFeed() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameImgRef = useRef<HTMLImageElement>(new Image());
  const tracks = useStore((s) => s.tracks);
  const telemetry = useStore((s) => s.telemetry);
  const frame = useStore((s) => s.frame);
  const playing = useStore((s) => s.playing);
  const setPlaying = useStore((s) => s.setPlaying);
  const now = useStore((s) => s.now);

  // "Draw concept" fetches the hi-res still *first*, then shows it — pausing
  // freezes frame N on screen, but the hi-res endpoint hands back whatever
  // frame arrives next, so fetching first is what makes the frame the
  // operator sees and the frame they draw on the same one. setPlaying(false)
  // still fires, but only to stop the live canvas updating behind the modal.
  const [context, setContext] = useState<{ imageSrc: string; imageBlob: Blob } | null>(null);
  const [capturing, setCapturing] = useState(false);
  const [captureError, setCaptureError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const [detectFps, setDetectFps] = useState(0);
  const frameCountRef = useRef(0);
  const detectCountRef = useRef(0);

  // The render loop must not restart when tracks change (that would reset the
  // animation and defeat time-based interpolation), so it reads tracks and
  // their arrival time through refs rather than closing over the values.
  const tracksRef = useRef(tracks);
  const tracksAtRef = useRef(performance.now());
  useEffect(() => {
    tracksRef.current = tracks;
    tracksAtRef.current = performance.now();
    detectCountRef.current += 1;
  }, [tracks]);

  // decode incoming JPEG data URLs off the render loop — the <img> just
  // holds whatever finished decoding most recently, draw() below reads it
  // fresh every frame regardless of exactly when it finishes loading
  useEffect(() => {
    if (frame) {
      frameImgRef.current.src = frame;
      frameCountRef.current += 1;
    }
  }, [frame]);

  // Both rates are measured, not assumed — display and detection run at
  // deliberately different rates (see backend/app/pipeline.py), so showing
  // only one number invites confusing the two.
  useEffect(() => {
    const id = setInterval(() => {
      setFps(frameCountRef.current);
      setDetectFps(detectCountRef.current);
      frameCountRef.current = 0;
      detectCountRef.current = 0;
    }, 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    let raf = 0;
    let t = 0;
    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== rect.width * dpr) {
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const ageS = (performance.now() - tracksAtRef.current) / 1000;
      render(ctx, rect.width, rect.height, tracksRef.current, t, frameImgRef.current, ageS);
      t += 1;
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
    // Empty deps on purpose: this loop runs for the component's lifetime and
    // reads live values through refs. Adding `tracks` here would tear down and
    // recreate the RAF loop on every detection message.
  }, []);

  return (
    <div className="flex h-full flex-col gap-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Button size="sm" variant="secondary" onClick={() => setPlaying(!playing)}>
            {playing ? <Pause /> : <Play />}
            {playing ? "Pause" : "Resume"}
          </Button>
          <Button
            size="sm"
            variant="outline"
            disabled={capturing}
            onClick={async () => {
              setPlaying(false);
              setCapturing(true);
              setCaptureError(null);
              try {
                const dataUrl = await api.fetchHiresFrame();
                const imageBlob = await (await fetch(dataUrl)).blob();
                setContext({ imageSrc: dataUrl, imageBlob });
              } catch (e) {
                setCaptureError(e instanceof Error ? e.message : "failed to capture frame");
              } finally {
                setCapturing(false);
              }
            }}
          >
            <Crosshair /> {capturing ? "Capturing..." : "Draw concept"}
          </Button>
          <Badge variant="info" className="gap-1">
            <Layers className="h-3 w-3" /> Tier 1 · SAM 3 overlays
          </Badge>
          {captureError && <span className="text-[11px] text-alert">{captureError}</span>}
        </div>
        <div className="font-mono text-xs text-muted-foreground">
          {fmtClock(now)} · {fps} fps video · {detectFps} fps detect · {tracks.length} tracks
        </div>
      </div>

      <div className="relative flex-1 overflow-hidden rounded-lg border border-border bg-black">
        <canvas ref={canvasRef} className="h-full w-full" />

        {/* HUD */}
        <div className="pointer-events-none absolute left-3 top-3 rounded bg-black/45 px-2.5 py-1.5 font-mono text-[11px] leading-relaxed text-emerald-300 backdrop-blur">
          <div>UAV-01 · EO/IR</div>
          <div>LAT {telemetry.lat.toFixed(5)}</div>
          <div>LON {telemetry.lon.toFixed(5)}</div>
          <div>ALT {telemetry.alt.toFixed(0)}m · HDG {telemetry.hdg.toFixed(0)}°</div>
        </div>
        <div className="pointer-events-none absolute right-3 top-3 flex items-center gap-1.5 rounded bg-black/45 px-2 py-1 text-[11px] font-medium text-red-400 backdrop-blur">
          <span className="h-2 w-2 animate-pulse-dot rounded-full bg-red-500" /> REC
        </div>

      </div>

      {context && (
        <ContextAuthorDialog
          imageSrc={context.imageSrc}
          imageBlob={context.imageBlob}
          onClose={() => {
            setContext(null);
            setPlaying(true);
          }}
        />
      )}
    </div>
  );
}
