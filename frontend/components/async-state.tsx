"use client";

import type { ReactNode } from "react";

export function AsyncState({
  loading,
  error,
  children,
}: {
  loading: boolean;
  error: string | null;
  children: ReactNode;
}) {
  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading catalog…</p>;
  }
  if (error) {
    return (
      <p role="alert" className="text-sm text-destructive">
        {error}
      </p>
    );
  }
  return children;
}
