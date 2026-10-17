import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { proxyGrading } from '@/lib/grading-proxy';

export async function GET(request: Request) {
  const started = performance.now();
  const sessionToken = (await cookies()).get('session_token')?.value;
  if (!sessionToken) return NextResponse.json({ progress: {} });
  return proxyGrading(request, '/progress', { started, sessionToken, transform: body => ({ progress: body }) });
}
