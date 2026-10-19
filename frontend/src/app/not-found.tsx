import { ButtonLink, SectionHeading } from "@/components/ui";

export default function NotFound() {
  return (
    <div className="flex flex-col items-center py-28">
      <SectionHeading as="h1" eyebrow="404" title="This page isn't here." description="It may have moved, or the link may be wrong." />
      <div className="mt-10 flex flex-wrap justify-center gap-2.5">
        <ButtonLink href="/redact" variant="primary" size="lg">
          New redaction
        </ButtonLink>
        <ButtonLink href="/jobs" size="lg">
          All jobs
        </ButtonLink>
      </div>
    </div>
  );
}
