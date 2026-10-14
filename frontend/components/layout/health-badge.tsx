"use client";

import { useEffect, useState } from "react";

import { getHealth } from "@/lib/api";

export function HealthBadge() {
  const [label, setLabel] = useState("API …");

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((health) => {
        if (!cancelled) {
          setLabel(`API ${health.api_version}`);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setLabel("API offline");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return <p className="font-mono text-[11px] tracking-wide text-brand">{label}</p>;
}
