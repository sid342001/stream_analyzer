import { useEffect, useRef, useState } from "react";
import { Pause, Play, Crosshair, Plus, Layers } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { useStore } from "@/store/useStore";
import { fmtClock } from "@/lib/utils";
import type { TrackedObject } from "@/lib/types";

/** Draws the live UAV frame + SAM-3 style overlays onto a canvas. */
function render(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  tracks: TrackedObject[],
  t: number,
  frameImg: HTMLImageElement | null,
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
  for (const o of tracks) {
    const x = o.x * w;
    const y = o.y * h;
    const bw = o.w * w;
    const bh = o.h * h;

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
  const addConcept = useStore((s) => s.addConcept);
  const now = useStore((s) => s.now);

  const [drawing, setDrawing] = useState(false);
  const [newName, setNewName] = useState("");
  const [fps, setFps] = useState(0);
  const frameCountRef = useRef(0);

  // decode incoming JPEG data URLs off the render loop — the <img> just
  // holds whatever finished decoding most recently, draw() below reads it
  // fresh every frame regardless of exactly when it finishes loading
  useEffect(() => {
    if (frame) {
      frameImgRef.current.src = frame;
      frameCountRef.current += 1;
    }
  }, [frame]);

  // measured, not assumed — counts actual "frame" messages received per
  // second, so this reflects real stream/network health (backend/app/config.py's
  // DISPLAY_FPS is only a target)
  useEffect(() => {
    const id = setInterval(() => {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
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
      render(ctx, rect.width, rect.height, tracks, t, frameImgRef.current);
      t += 1;
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [tracks]);

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
            variant={drawing ? "default" : "outline"}
            onClick={() => {
              setPlaying(false);
              setDrawing((d) => !d);
            }}
          >
            <Crosshair /> Draw concept
          </Button>
          <Badge variant="info" className="gap-1">
            <Layers className="h-3 w-3" /> Tier 1 · SAM 3 overlays
          </Badge>
        </div>
        <div className="font-mono text-xs text-muted-foreground">
          {fmtClock(now)} · {fps} fps · {tracks.length} tracks
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

        {drawing && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/30">
            <div className="w-72 animate-fade-in rounded-lg border border-border bg-panel p-4 shadow-2xl">
              <div className="mb-1 text-sm font-semibold">Teach a new concept</div>
              <p className="mb-3 text-xs text-muted-foreground">
                Frame paused. Name it and SAM 3 starts looking for it by text —
                drawing example boxes for a tighter match isn't wired up yet.
              </p>
              <Input
                autoFocus
                placeholder="e.g. checkpoint, boat, cook-fire"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                className="mb-3"
              />
              <div className="flex justify-end gap-2">
                <Button size="sm" variant="ghost" onClick={() => setDrawing(false)}>
                  Cancel
                </Button>
                <Button
                  size="sm"
                  disabled={!newName.trim()}
                  onClick={() => {
                    addConcept(newName.trim());
                    setNewName("");
                    setDrawing(false);
                    setPlaying(true);
                  }}
                >
                  <Plus /> Add & go live
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
