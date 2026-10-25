// Small shared UI pieces.

import type { ReactNode } from "react";
import type { Compound, Provenance } from "@/lib/api";
import { COMPOUND_COLOR } from "@/lib/race";

export function Panel({ title, actions, children, className = "" }: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-lg border border-line bg-surface-1 ${className}`}>
      {(title || actions) && (
        <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
          <h2 className="text-sm font-semibold text-ink">{title}</h2>
          {actions}
        </div>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

/** Where numbers come from: solid = what happened, dashed = simulated estimate. */
export function ProvenanceBadge({ kind }: { kind: Provenance }) {
  const simulated = kind === "simulated";
  return (
    <span
      title={simulated ? "Counterfactual estimate from the race simulator" : "Recorded race data"}
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ${
        simulated ? "border border-dashed border-focus text-focus" : "border border-line-strong text-ink-2"
      }`}
    >
      {simulated ? "Simulated" : "Historical"}
    </span>
  );
}

export function CompoundChip({ compound, size = "md" }: { compound: Compound; size?: "sm" | "md" }) {
  const d = size === "sm" ? "h-4 w-4 text-[9px]" : "h-5 w-5 text-[10px]";
  return (
    <span
      aria-label={compound.toLowerCase()}
      className={`inline-flex ${d} items-center justify-center rounded-full border-2 font-bold text-ink`}
      style={{ borderColor: COMPOUND_COLOR[compound] }}
    >
      {compound[0]}
    </span>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wide text-ink-3">{label}</div>
      <div className="tabular text-lg font-semibold text-ink">{value}</div>
      {sub && <div className="tabular text-xs text-ink-2">{sub}</div>}
    </div>
  );
}

export function Button({ children, onClick, disabled, variant = "primary", type = "button", title }: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "ghost";
  type?: "button" | "submit";
  title?: string;
}) {
  const style =
    variant === "primary"
      ? "bg-focus text-white hover:brightness-110 disabled:opacity-40"
      : "border border-line-strong text-ink-2 hover:text-ink hover:border-ink-3 disabled:opacity-40";
  return (
    <button type={type} onClick={onClick} disabled={disabled} title={title}
      className={`rounded-md px-3 py-1.5 text-sm font-medium transition ${style} disabled:cursor-not-allowed`}>
      {children}
    </button>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div role="alert" className="rounded-md border border-warn/50 bg-warn/10 px-3 py-2 text-sm text-ink">
      {message}
    </div>
  );
}
