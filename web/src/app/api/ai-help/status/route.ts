import { NextResponse } from 'next/server';

export async function GET() {
  const configured = Boolean(
    process.env.AI_HELP_BASE_URL?.trim() &&
    process.env.AI_HELP_API_KEY?.trim() &&
    process.env.AI_HELP_MODEL?.trim(),
  );
  return NextResponse.json({ configured });
}
