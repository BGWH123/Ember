import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { GRADING_SERVICE_URL } from '@/lib/constants';

export async function GET(_request: Request, { params }: { params: Promise<{ submissionId: string }> }) {
  const { submissionId } = await params;
  const sessionToken = (await cookies()).get('session_token')?.value;
  if (!sessionToken) return NextResponse.json({ error: 'Session not found' }, { status: 401 });
  const response = await fetch(`${GRADING_SERVICE_URL}/submissions/${submissionId}`, {
    headers: { 'X-Session-Token': sessionToken }, cache: 'no-store',
  });
  return NextResponse.json(await response.json(), { status: response.status });
}
