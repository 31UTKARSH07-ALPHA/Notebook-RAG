import { NextResponse } from "next/server";
import evalDataset from "@/lib/eval-dataset.json";

export async function GET() {
  return NextResponse.json(evalDataset);
}
