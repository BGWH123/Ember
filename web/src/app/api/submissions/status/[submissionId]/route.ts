import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { proxyGrading } from '@/lib/grading-proxy';

export async function GET(request: Request, { params }: { params: Promise<{ submissionId: string }> }) {
  const started = performance.now();
  const { submissionId } = await params;
  const sessionToken = (await cookies()).get('session_token')?.value;
  if (!sessionToken) return NextResponse.json({ error: 'Session not found' }, { status: 401 });
  return proxyGrading(request, `/submissions/${encodeURIComponent(submissionId)}`, { started, sessionToken });
}
