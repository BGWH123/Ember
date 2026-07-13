import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { GRADING_SERVICE_URL } from '@/lib/constants';

export async function GET(_req: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const { taskId } = await params;
  const sessionToken = (await cookies()).get('session_token')?.value;
  if (!sessionToken) return NextResponse.json([]);
  const response = await fetch(`${GRADING_SERVICE_URL}/submissions/history/${taskId}`, { headers: { 'X-Session-Token': sessionToken }, cache: 'no-store' });
  return NextResponse.json(response.ok ? await response.json() : []);
}
