/* Small pieces shared by every page. Server strings are rendered as React text, never as HTML.

   The source project's StampCard is deliberately absent: this API returns no provenance envelope
   at all (docs/BASELINE.md), so there is nothing to put in one. Provenance comes from the
   read-only endpoints added for it, and lives in src/model/pages.tsx. */

import type { ReactNode } from "react";
import { ApiError } from "../api/client";

/** A value that `toFixed(d)` would show as zero, shown as whichever of plain decimal or scientific
 *  notation is shorter (4 significant digits, trailing zeros dropped). Never rounds to zero. */
function tiny(v: number): string {
  const sci = v.toExponential(3).replace(/\.?0+e/, "e").replace("e+", "e");
  const sig = Number(v.toPrecision(4));
  const places = Math.min(100, Math.max(0, 3 - Math.floor(Math.log10(Math.abs(sig)))));
  const plain = sig.toFixed(places).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
  return plain.length <= sci.length ? plain : sci;
}

/** Fixed `d` decimals, except a nonzero value that would display as 0 (e.g. 0.0004 at 2 dp). */
export const num = (v: number | null | undefined, d = 1): string | null => {
  if (v == null) return null;
  const n = Number(v);
  const fixed = n.toFixed(d);
  return n !== 0 && Number.isFinite(n) && Number(fixed) === 0 ? tiny(n) : fixed;
};
export const pct = (v: number | null | undefined): string | null => {
  const n = num(v == null ? v : v * 100, 2);
  return n == null ? null : n + "%";
};
export const words = (s: string | null | undefined): string => String(s ?? "").replace(/_/g, " ");

export function Metric({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="metric">
      <small>{label}</small>
      <b>{value ?? "not defined"}</b>
    </div>
  );
}

/** `plain` for values a person reads (an email, a date); the default keeps the source's
 *  break-all, which suits an opaque id and shreds anything else. */
export function Kv({ k, v, plain = false }: { k: string; v: ReactNode; plain?: boolean }) {
  return (
    <>
      <dt>{k}</dt>
      <dd className={plain ? "mono plain" : "mono"}>{v}</dd>
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
