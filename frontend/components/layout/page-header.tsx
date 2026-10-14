import type { ReactNode } from "react";

type PageHeaderProps = {
  label?: string;
  title: string;
  description?: string;
  actions?: ReactNode;
};

export function PageHeader({ label, title, description, actions }: PageHeaderProps) {
  return (
    <header className="mb-8 flex flex-col gap-4 border-b border-border/60 pb-7 sm:flex-row sm:items-end sm:justify-between">
      <div className="flex min-w-0 gap-3">
        <div className="mt-1.5 hidden h-9 w-1 shrink-0 rounded-full bg-brand sm:block" aria-hidden />
        <div className="min-w-0">
          {label ? <p className="label-caps">{label}</p> : null}
          <h1 className="text-display text-2xl font-semibold sm:text-3xl">{title}</h1>
          {description ? (
            <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">{description}</p>
          ) : null}
        </div>
      </div>
      {actions ? <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div> : null}
    </header>
  );
}
