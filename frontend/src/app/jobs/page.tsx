import type { Metadata } from "next";
import { Plus } from "lucide-react";

import { JobsTable } from "@/components/jobs-table";
import { ButtonLink, PageTitle } from "@/components/ui";

export const metadata: Metadata = { title: "Jobs" };

export default function JobsPage() {
  return (
    <>
      <PageTitle
        eyebrow="Jobs"
        title={
          <>
            Every run.
            <br /> On the record.
          </>
        }
        description="Every redaction run on the server. Inputs are deleted when a job ends; the artefacts stay until you delete the job."
        actions={
          <ButtonLink href="/redact" variant="primary" size="lg">
            <Plus className="size-4" aria-hidden />
            New redaction
          </ButtonLink>
        }
      />
      <JobsTable />
    </>
  );
}
