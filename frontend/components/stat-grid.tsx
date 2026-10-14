type StatItem = {
  label: string;
  value: string;
};

export function StatGrid({ items }: { items: StatItem[] }) {
  return (
    <section className="grid grid-cols-2 gap-3">
      {items.map((item) => (
        <div key={item.label} className="surface-raised flex flex-col gap-1 px-4 py-3">
          <span className="label-caps">{item.label}</span>
          <strong className="font-mono text-sm tabular-nums">{item.value}</strong>
        </div>
      ))}
    </section>
  );
}
