import { NextResponse } from "next/server";
import { readDemoConfig } from "@/lib/demo-limits";

export const runtime = "nodejs";
// Read env at request time, not at build time, so DEMO_MODE can change without a rebuild
export const dynamic = "force-dynamic";

// Tells the UI whether to show the public-demo note and which limits to mention.
export async function GET() {
  const config = readDemoConfig();
  if (!config.enabled) return NextResponse.json({ demoMode: false });
  return NextResponse.json({
    demoMode: true,
    maxUploadsPerDay: config.maxUploadsPerDay,
    maxQuestionsPerDay: config.maxQuestionsPerDay,
    notebookTtlHours: config.notebookTtlHours,
  });
}
