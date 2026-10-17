import {
  Accessibility,
  Barcode,
  CalendarDays,
  Car,
  CreditCard,
  HeartPulse,
  Landmark,
  MapPin,
  Medal,
  PenLine,
  Receipt,
  ScanFace,
  Stamp,
  Stethoscope,
  User,
  type LucideIcon,
} from "lucide-react";

import { cx } from "./ui";

/** A sample of what the engine finds, alternating serif and bold sans like a row of wordmarks. */
const ITEMS: { icon: LucideIcon; label: string }[] = [
  { icon: User, label: "Person" },
  { icon: CreditCard, label: "Medicare number" },
  { icon: HeartPulse, label: "IHI" },
  { icon: Stethoscope, label: "AHPRA" },
  { icon: Medal, label: "DVA file" },
  { icon: Landmark, label: "Centrelink CRN" },
  { icon: Receipt, label: "Tax file number" },
  { icon: Accessibility, label: "NDIS" },
  { icon: Car, label: "Driver licence" },
  { icon: CalendarDays, label: "Date of birth" },
  { icon: MapPin, label: "Street address" },
  { icon: PenLine, label: "Signature" },
  { icon: ScanFace, label: "Face" },
  { icon: Barcode, label: "Barcode" },
  { icon: Stamp, label: "Stamp" },
];

function Track({ hidden }: { hidden?: boolean }) {
  return (
    <ul className="flex shrink-0 items-center gap-14 pr-14" aria-hidden={hidden || undefined}>
      {ITEMS.map(({ icon: Icon, label }, i) => (
        <li key={label} className="flex items-center gap-2.5 whitespace-nowrap text-ink-2/80 transition-colors duration-300 hover:text-ink">
          <Icon className="size-6" strokeWidth={i % 2 ? 2.4 : 1.75} aria-hidden />
          <span
            className={cx(
              "text-[1.75rem] leading-none",
              i % 2 ? "font-sans font-bold tracking-[-0.03em]" : "font-serif",
            )}
          >
            {label}
          </span>
        </li>
      ))}
    </ul>
  );
}

export function IdentifierMarquee({ caption }: { caption: string }) {
  return (
    <section aria-label="Identifiers the engine detects" data-reveal className="py-16">
      <p className="text-center text-[0.9375rem] text-ink-2">{caption}</p>
      <div className="marquee mt-9 overflow-hidden">
        <div className="marquee-track flex w-max">
          <Track />
          <Track hidden />
        </div>
      </div>
    </section>
  );
}
