import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { proxyGrading } from '@/lib/grading-proxy';

export async function GET(request: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const started = performance.now();
  const { taskId } = await params;
  const sessionToken = (await cookies()).get('session_token')?.value;
  if (!sessionToken) return NextResponse.json([]);
  return proxyGrading(request, `/submissions/history/${encodeURIComponent(taskId)}`, { started, sessionToken });
}
