import { useRef, useState } from "react";
import { AlertTriangle, ChevronRight, Library, Plus, Radio } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ContextAuthorDialog } from "@/components/ContextAuthorDialog";
import { useStore } from "@/store/useStore";
import { cn } from "@/lib/utils";
import * as api from "@/lib/api";
import type { Priority } from "@/lib/types";

const priorityBadge: Record<Priority, "alert" | "watch" | "default"> = {
  alert: "alert",
  watch: "watch",
  ignore: "default",
};

export function LeftRail() {
  const streamStatus = useStore((s) => s.streamStatus);
  const concepts = useStore((s) => s.concepts);
  const tracks = useStore((s) => s.tracks);
  const toggleConcept = useStore((s) => s.toggleConcept);
  const cyclePriority = useStore((s) => s.cycleConceptPriority);

  // Backend has a single STREAM_URL (backend/app/config.py) — this is
  // runtime settings for that one source, not a multi-stream manager.
  const [host, setHost] = useState("0.0.0.0");
  const [port, setPort] = useState("5600");
  const [connecting, setConnecting] = useState(false);
  const [connectError, setConnectError] = useState<string | null>(null);

  async function handleConnect() {
    const portNum = Number(port);
    if (!host.trim() || !Number.isInteger(portNum) || portNum <= 0 || portNum > 65535) {
      setConnectError("enter a valid host and port");
      return;
    }
    setConnecting(true);
    setConnectError(null);
    try {
      await api.setStream(host.trim(), portNum);
      // Outcome (connected/error) arrives async via the WS `status`
      // message — see App.tsx's onStatus wiring — not this response.
    } catch (e) {
      setConnectError(e instanceof Error ? e.message : "failed to reconnect");
    } finally {
      setConnecting(false);
    }
  }

  // Client-side only — the store already has everything needed, no backend
  // round-trip. Matches on label since TrackedObject.concept is populated
  // from Concept.label (pipeline._broadcast_tracks); two concepts sharing a
  // label would show the combined count, flagged below rather than silently
  // miscounted.
  const liveCounts = new Map<string, number>();
  for (const t of tracks) liveCounts.set(t.concept, (liveCounts.get(t.concept) ?? 0) + 1);
  const labelCounts = new Map<string, number>();
  for (const c of concepts) labelCounts.set(c.label, (labelCounts.get(c.label) ?? 0) + 1);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadContext, setUploadContext] = useState<{ imageSrc: string; imageBlob: Blob } | null>(
    null,
  );

  function handleFileChosen(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow re-selecting the same file later
    if (!file) return;
    setUploadContext({ imageSrc: URL.createObjectURL(file), imageBlob: file });
  }

  function closeUploadContext() {
    if (uploadContext) URL.revokeObjectURL(uploadContext.imageSrc);
    setUploadContext(null);
  }

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-border bg-panel">
      <div className="p-3">
        <div className="mb-2 flex items-center justify-between">
          <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
            Stream
          </div>
          <div className="flex items-center gap-1.5">
            <span
              className={cn(
                "h-1.5 w-1.5 rounded-full",
                streamStatus?.connected ? "bg-online animate-pulse-dot" : "bg-muted-foreground",
              )}
            />
            <span className="text-[10px] text-muted-foreground">
              {streamStatus === null
                ? "unknown"
                : streamStatus.connected
                  ? "connected"
                  : streamStatus.error
                    ? "disconnected"
                    : "reconnecting…"}
            </span>
          </div>
        </div>
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-1.5">
            <Radio className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
            <Input
              value={host}
              onChange={(e) => setHost(e.target.value)}
              placeholder="0.0.0.0"
              className="h-7 flex-1 font-mono text-xs"
            />
            <span className="text-muted-foreground">:</span>
            <Input
              value={port}
              onChange={(e) => setPort(e.target.value)}
              placeholder="5600"
              className="h-7 w-16 font-mono text-xs"
            />
          </div>
          <Button size="xs" variant="outline" onClick={handleConnect} disabled={connecting}>
            {connecting ? "Connecting…" : "Connect"}
          </Button>
          <p className="text-[10px] leading-snug text-muted-foreground">
            Address the backend listens on for incoming UDP — not a peer it connects out to.
          </p>
          {(connectError || streamStatus?.error) && (
            <p className="text-[10px] text-alert">{connectError ?? streamStatus?.error}</p>
          )}
        </div>
      </div>

      <Separator />

      <div className="flex items-center justify-between px-3 pt-3">
        <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
          Watch-list
        </div>
        <Badge variant="outline" className="text-[10px]">{concepts.filter((c) => c.enabled).length} live</Badge>
      </div>

      <ScrollArea className="flex-1 px-2 py-2">
        <div className="flex flex-col gap-1">
          {concepts.map((c) => (
            <div
              key={c.id}
              className="flex items-center gap-2 rounded-md px-1.5 py-1.5 hover:bg-accent/60"
            >
              <span className="h-3 w-3 shrink-0 rounded-sm" style={{ background: c.color }} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1">
                  <div className="truncate text-xs font-medium capitalize">{c.label}</div>
                  {(labelCounts.get(c.label) ?? 0) > 1 && (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <AlertTriangle className="h-3 w-3 shrink-0 text-watch" />
                      </TooltipTrigger>
                      <TooltipContent>
                        Multiple concepts share this label — the live count below is combined.
                      </TooltipContent>
                    </Tooltip>
                  )}
                </div>
                <div className="text-[10px] text-muted-foreground">
                  {c.exemplars} exemplars · {liveCounts.get(c.label) ?? 0} active now
                </div>
              </div>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button onClick={() => cyclePriority(c.id)}>
                    <Badge variant={priorityBadge[c.priority]} className="cursor-pointer text-[10px] capitalize">
                      {c.priority}
                    </Badge>
                  </button>
                </TooltipTrigger>
                <TooltipContent>Click to cycle priority</TooltipContent>
              </Tooltip>
              <Switch checked={c.enabled} onCheckedChange={() => toggleConcept(c.id)} />
            </div>
          ))}
        </div>
      </ScrollArea>

      <div className="flex flex-col gap-2 border-t border-border p-3">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={handleFileChosen}
        />
        <Button variant="outline" size="sm" className="justify-start" onClick={() => fileInputRef.current?.click()}>
          <Plus /> Add context
        </Button>
        <Button variant="ghost" size="sm" className="justify-between text-muted-foreground">
          <span className="flex items-center gap-2">
            <Library /> Concept library
          </span>
          <ChevronRight className="h-3.5 w-3.5" />
        </Button>
      </div>

      {uploadContext && (
        <ContextAuthorDialog
          imageSrc={uploadContext.imageSrc}
          imageBlob={uploadContext.imageBlob}
          onClose={closeUploadContext}
        />
      )}
    </aside>
  );
}
