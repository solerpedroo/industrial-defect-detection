import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

export function FormPanel({
  title,
  subtitle,
  children,
  className,
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("surface-raised overflow-hidden", className)}>
      {title ? (
        <header className="border-b border-border/70 px-5 py-4">
          <h2 className="text-sm font-semibold">{title}</h2>
          {subtitle ? <p className="mt-1 text-xs leading-5 text-muted-foreground">{subtitle}</p> : null}
        </header>
      ) : null}
      <div className="p-5">{children}</div>
    </section>
  );
}
