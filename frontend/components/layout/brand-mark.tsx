import { cn } from "@/lib/utils";

type BrandMarkProps = {
  className?: string;
  size?: "sm" | "md";
};

export function BrandMark({ className, size = "md" }: BrandMarkProps) {
  const box = size === "sm" ? "size-8" : "size-9";
  const icon = size === "sm" ? 16 : 18;

  return (
    <span
      className={cn(
        "flex shrink-0 items-center justify-center rounded-lg border border-brand/25 bg-brand/10 text-brand",
        box,
        className,
      )}
      aria-hidden
    >
      <svg width={icon} height={icon} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="4" y="4" width="16" height="16" rx="3" stroke="currentColor" strokeWidth="1.75" />
        <circle cx="12" cy="12" r="2.25" fill="currentColor" />
        <path d="M8 15.5h8" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" />
      </svg>
    </span>
  );
}
