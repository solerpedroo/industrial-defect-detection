"use client";

import { FlaskConical, FolderKanban, GitCompare, LayoutDashboard, ScanSearch } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { PRODUCT_NAME, PRODUCT_TAGLINE } from "@/lib/brand";
import { cn } from "@/lib/utils";

import { BrandMark } from "./brand-mark";
import { HealthBadge } from "./health-badge";

export const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/datasets", label: "Datasets", icon: FolderKanban },
  { href: "/inference", label: "Inference Lab", icon: ScanSearch },
  { href: "/experiments", label: "Experiments", icon: FlaskConical },
  { href: "/comparison", label: "Comparison", icon: GitCompare },
] as const;

export function isActivePath(pathname: string, href: string) {
  return href === "/" ? pathname === "/" : pathname === href || pathname.startsWith(`${href}/`);
}

export function AppSidebar() {
  const pathname = usePathname();

  return (
    <aside className="sidebar-chrome fixed inset-y-0 left-0 z-40 hidden w-[260px] flex-col border-r border-sidebar-border lg:flex">
      <div className="flex h-16 items-center gap-2.5 border-b border-sidebar-border px-5">
        <BrandMark />
        <div className="min-w-0 leading-tight">
          <p className="truncate text-sm font-semibold tracking-tight">{PRODUCT_NAME}</p>
          <p className="truncate text-[11px] text-muted-foreground">{PRODUCT_TAGLINE}</p>
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-3 py-4" aria-label="Primary">
        <p className="sidebar-section-label px-3 pb-2">Workspace</p>
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = isActivePath(pathname, href);
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? "page" : undefined}
              className={cn(
                "relative flex items-center gap-2.5 rounded-lg px-3 py-2.5 text-[13px] transition-colors",
                active
                  ? "nav-active font-medium text-foreground"
                  : "text-muted-foreground hover:bg-sidebar-accent hover:text-foreground",
              )}
            >
              <Icon className={cn("size-4", active && "text-brand")} />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="space-y-3 border-t border-sidebar-border p-4">
        <HealthBadge />
        <p className="text-xs leading-5 text-muted-foreground">
          Training runs offline. This API serves catalogs and, later, prepared checkpoints.
        </p>
      </div>
    </aside>
  );
}
