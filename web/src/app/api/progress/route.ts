import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { GRADING_SERVICE_URL } from '@/lib/constants';

export async function GET() {
  const sessionToken = (await cookies()).get('session_token')?.value;
  if (!sessionToken) return NextResponse.json({ progress: {} });
  const response = await fetch(`${GRADING_SERVICE_URL}/progress`, { headers: { 'X-Session-Token': sessionToken }, cache: 'no-store' });
  return NextResponse.json({ progress: response.ok ? await response.json() : {} });
}
