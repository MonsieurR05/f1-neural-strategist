"use client";

import { useEffect, useRef, useState } from "react";

/** Width of an element, kept up to date with a ResizeObserver (for responsive SVG charts). */
export function useWidth<T extends HTMLElement>(initial = 600) {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(240, entry.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}

/** Seconds elapsed while ``running`` is true (for long requests). */
export function useElapsed(running: boolean) {
  const [elapsed, setElapsed] = useState(0);
  const started = useRef(0);
  useEffect(() => {
    if (!running) return;
    started.current = performance.now();
    const id = setInterval(() => setElapsed((performance.now() - started.current) / 1000), 250);
    return () => {
      clearInterval(id);
      setElapsed(0);
    };
  }, [running]);
  return elapsed;
}
