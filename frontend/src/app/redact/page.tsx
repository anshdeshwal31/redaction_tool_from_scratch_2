import type { Metadata } from "next";

import { NewJobForm } from "@/components/new-job-form";
import { PageTitle } from "@/components/ui";

export const metadata: Metadata = { title: "New redaction" };

export default function RedactPage() {
  return (
    <>
      <PageTitle
        eyebrow="New redaction"
        title={
          <>
            One document in.
            <br /> One rebuilt PDF out.
          </>
        }
        description="It is released only if every verification check passes. Otherwise it is quarantined with content-free reason codes."
      />
      <NewJobForm />
    </>
  );
}
