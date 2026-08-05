import { useCallback, useEffect, useRef, useState } from "react";

interface Props {
  imageSrc: string;
  onBoxChange: (box: [number, number, number, number] | null) => void;
}

/** Static (non-RAF) canvas for drawing a rubber-band box on a still image —
 *  deliberately separate from CenterFeed's live canvas, which runs its own
 *  requestAnimationFrame loop built specifically to avoid render coupling;
 *  bolting drag state into that loop would mean threading another ref
 *  through code that goes out of its way not to need one.
 *
 *  Emits the box in the *image's own natural pixel coordinates*, not canvas
 *  CSS pixels — scale factors are computed separately for X and Y since the
 *  canvas can stretch without preserving the image's aspect ratio.
 */
export function BoxDrawOverlay({ imageSrc, onBoxChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(new Image());
  const [ready, setReady] = useState(false);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);
  const [boxCss, setBoxCss] = useState<[number, number, number, number] | null>(null);

  useEffect(() => {
    setReady(false);
    setBoxCss(null);
    onBoxChange(null);
    const img = imgRef.current;
    img.onload = () => setReady(true);
    img.src = imageSrc;
    // onBoxChange intentionally omitted: it's a setter passed fresh every
    // render by the parent, including it would re-run this on every parent
    // render and reset the drawn box for no reason.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageSrc]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !ready) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.drawImage(img, 0, 0, rect.width, rect.height);
    if (boxCss) {
      const [x1, y1, x2, y2] = boxCss;
      ctx.fillStyle = "#38bdf822";
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
  }, [ready, boxCss]);

  useEffect(() => draw(), [draw]);
  useEffect(() => {
    window.addEventListener("resize", draw);
    return () => window.removeEventListener("resize", draw);
  }, [draw]);

  function toCanvasPoint(e: React.PointerEvent<HTMLCanvasElement>) {
    const rect = canvasRef.current!.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  function handlePointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    const p = toCanvasPoint(e);
    dragStartRef.current = p;
    setBoxCss([p.x, p.y, p.x, p.y]);
    e.currentTarget.setPointerCapture(e.pointerId);
  }

  function handlePointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    const s = dragStartRef.current;
    if (!s) return;
    const p = toCanvasPoint(e);
    setBoxCss([Math.min(s.x, p.x), Math.min(s.y, p.y), Math.max(s.x, p.x), Math.max(s.y, p.y)]);
  }

  function handlePointerUp(e: React.PointerEvent<HTMLCanvasElement>) {
    const s = dragStartRef.current;
    dragStartRef.current = null;
    if (!s) return;
    const p = toCanvasPoint(e);
    const finalCss: [number, number, number, number] = [
      Math.min(s.x, p.x),
      Math.min(s.y, p.y),
      Math.max(s.x, p.x),
      Math.max(s.y, p.y),
    ];
    setBoxCss(finalCss);

    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img.naturalWidth || finalCss[2] - finalCss[0] < 4 || finalCss[3] - finalCss[1] < 4) {
      onBoxChange(null);
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;
    onBoxChange([
      Math.round(finalCss[0] * scaleX),
      Math.round(finalCss[1] * scaleY),
      Math.round(finalCss[2] * scaleX),
      Math.round(finalCss[3] * scaleY),
    ]);
  }

  return (
    <canvas
      ref={canvasRef}
      className="h-full w-full cursor-crosshair touch-none"
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    />
  );
}
