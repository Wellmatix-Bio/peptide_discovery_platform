/* Small pieces shared by every page. Server strings are rendered as React text, never as HTML.

   The source project's StampCard is deliberately absent: this API returns no provenance envelope
   at all (docs/BASELINE.md), so there is nothing to put in one. Provenance comes from the
   read-only endpoints added for it, and lives in src/model/pages.tsx. */

import type { ReactNode } from "react";
import { ApiError } from "../api/client";

export const pct = (v: number | null | undefined): string | null => (v == null ? null : (v * 100).toFixed(2) + "%");
export const num = (v: number | null | undefined, d = 1): string | null => (v == null ? null : Number(v).toFixed(d));
export const words = (s: string | null | undefined): string => String(s ?? "").replace(/_/g, " ");

export function Metric({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="metric">
      <small>{label}</small>
      <b>{value ?? "not defined"}</b>
    </div>
  );
}

export function Kv({ k, v }: { k: string; v: ReactNode }) {
  return (
    <>
      <dt>{k}</dt>
      <dd className="mono">{v}</dd>
    </>
  );
}

export function Pill({ tone, children }: { tone: "good" | "warn" | "bad" | "muted"; children: ReactNode }) {
  return <span className={`pill ${tone}`}>{children}</span>;
}

export function Hero({ title, sub, badge, badgeStyle }: { title: string; sub?: ReactNode; badge?: ReactNode; badgeStyle?: React.CSSProperties }) {
  return (
    <div className="hero">
      <div>
        <h1>{title}</h1>
        {sub ? <p>{sub}</p> : null}
      </div>
      {badge ? (
        <div className="badge" style={badgeStyle}>
          {badge}
        </div>
      ) : null}
    </div>
  );
}

export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null;
  const status = error instanceof ApiError ? error.status : 0;
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="err">
      <b>{status ? `HTTP ${status}` : "Request failed"}</b>
      {"\n"}
      {message}
    </div>
  );
}

export function Loading({ children = "Loading…" }: { children?: ReactNode }) {
  return <div className="loading">{children}</div>;
}

export function downloadJson(name: string, data: unknown): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}
