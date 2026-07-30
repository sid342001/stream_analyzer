/**
 * REST client for the backend's context-authoring API (backend/app/main.py).
 * Routed through vite's `/api` proxy in dev (see vite.config.ts) so this
 * works the same whether the backend runs on localhost or in a container.
 */
import type { Concept, Priority } from "./types";

const BASE = "/api";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
  return res.json() as Promise<T>;
}

export function fetchConcepts(): Promise<Concept[]> {
  return fetch(`${BASE}/concepts`).then(json<Concept[]>);
}

export function createConcept(label: string, priority: Priority = "watch"): Promise<Concept> {
  return fetch(`${BASE}/concepts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label, priority }),
  }).then(json<Concept>);
}

export function updateConcept(
  id: string,
  patch: { enabled?: boolean; priority?: Priority },
): Promise<Concept> {
  return fetch(`${BASE}/concepts/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  }).then(json<Concept>);
}
