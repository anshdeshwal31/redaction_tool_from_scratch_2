import type { Metadata } from "next";

import { JobDetail } from "@/components/job-detail";

type Props = { params: Promise<{ id: string }> };

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params;
  return { title: `Job ${id.slice(0, 8)}` };
}

export default async function JobPage({ params }: Props) {
  const { id } = await params;
  return <JobDetail id={id} />;
}
