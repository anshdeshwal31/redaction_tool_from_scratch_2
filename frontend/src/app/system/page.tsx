import type { Metadata } from "next";

import { SystemPanel } from "@/components/system-panel";
import { PageTitle } from "@/components/ui";

export const metadata: Metadata = { title: "System" };

export default function SystemPage() {
  return (
    <>
      <PageTitle
        eyebrow="System"
        title={
          <>
            The engine,
            <br /> checked and pinned.
          </>
        }
        description="The engine this interface drives, its configuration, and whether its models and resources verify."
      />
      <SystemPanel />
    </>
  );
}
