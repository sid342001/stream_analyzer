import { useState } from "react";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { BoxDrawOverlay } from "@/components/BoxDrawOverlay";
import { useStore } from "@/store/useStore";
import * as api from "@/lib/api";

interface Props {
  imageSrc: string;
  imageBlob: Blob;
  onClose: () => void;
}

/** Shared by both context-authoring entry points — CenterFeed's "Draw
 *  concept" (a hi-res still fetched from the live stream) and LeftRail's
 *  "Add context" (an uploaded file). Either way, by the time this renders
 *  the caller already has a concrete image + blob to draw on and submit;
 *  this component only owns the box-drawing and the concept name/submit UI.
 */
export function ContextAuthorDialog({ imageSrc, imageBlob, onClose }: Props) {
  const concepts = useStore((s) => s.concepts);
  const addConcept = useStore((s) => s.addConcept);
  const loadConcepts = useStore((s) => s.loadConcepts);
  const [name, setName] = useState("");
  const [box, setBox] = useState<[number, number, number, number] | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit() {
    const label = name.trim();
    if (!label || !box) return;
    setSubmitting(true);
    setError(null);
    try {
      const existing = concepts.find((c) => c.label.toLowerCase() === label.toLowerCase());
      const concept = existing ?? (await addConcept(label));
      await api.addExemplar(concept.id, imageBlob, box);
      await loadConcepts(); // pick up the bumped exemplar count
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "failed to add exemplar");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="flex max-h-[90vh] w-[min(90vw,900px)] flex-col gap-3 rounded-lg border border-border bg-panel p-4 shadow-2xl">
        <div>
          <div className="text-sm font-semibold">Teach a concept from this frame</div>
          <p className="text-xs text-muted-foreground">
            Drag a box around the object. DINOv3 embeds it as a reference exemplar — future
            detections only count as this concept if they're similar enough to what you draw here.
          </p>
        </div>

        <div className="relative aspect-video w-full overflow-hidden rounded-md border border-border bg-black">
          <BoxDrawOverlay imageSrc={imageSrc} onBoxChange={setBox} />
        </div>

        <div className="flex items-end gap-2">
          <div className="flex-1">
            <label className="mb-1 block text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
              Concept name
            </label>
            <Input
              autoFocus
              list="context-author-concept-labels"
              placeholder="e.g. checkpoint, boat, cook-fire"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
            <datalist id="context-author-concept-labels">
              {concepts.map((c) => (
                <option key={c.id} value={c.label} />
              ))}
            </datalist>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose} disabled={submitting}>
            Cancel
          </Button>
          <Button size="sm" disabled={!name.trim() || !box || submitting} onClick={handleSubmit}>
            <Plus /> {submitting ? "Adding..." : "Add exemplar"}
          </Button>
        </div>
        {!box && (
          <p className="text-[11px] text-muted-foreground">Draw a box on the image above to continue.</p>
        )}
        {error && <p className="text-[11px] text-alert">{error}</p>}
      </div>
    </div>
  );
}
