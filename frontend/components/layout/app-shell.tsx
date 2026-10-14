"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

import { AppSidebar, isActivePath, NAV_ITEMS } from "./app-sidebar";
import { BrandMark } from "./brand-mark";

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="relative min-h-screen">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-brand focus:px-3 focus:py-2 focus:text-sm focus:font-medium focus:text-brand-foreground"
      >
        Skip to main content
      </a>

      <AppSidebar />

      <div className="flex min-h-screen flex-col lg:pl-[260px]">
        <header className="sticky top-0 z-30 border-b border-border/70 bg-background/85 backdrop-blur lg:hidden">
          <div className="flex items-center gap-2 px-3 py-3">
            <BrandMark size="sm" />
            <p className="text-sm font-semibold">Defect Detection</p>
          </div>
          <nav className="flex gap-1 overflow-x-auto px-3 pb-3" aria-label="Primary mobile">
            {NAV_ITEMS.map((item) => {
              const active = isActivePath(pathname, item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "shrink-0 rounded-full px-3 py-1.5 text-xs",
                    active ? "bg-brand text-brand-foreground" : "text-muted-foreground",
                  )}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </header>

        <main id="main-content" tabIndex={-1} className="relative flex-1 px-3 py-6 sm:px-5 lg:px-8 lg:py-8">
          <div className="canvas-grid pointer-events-none absolute inset-x-0 top-0 h-72" aria-hidden />
          <div className="relative mx-auto w-full max-w-6xl">{children}</div>
        </main>
      </div>
    </div>
  );
}
