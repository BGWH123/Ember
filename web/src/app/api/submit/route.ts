import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { GRADING_SERVICE_URL } from '@/lib/constants';

export async function POST(request: Request) {
  const payload = await request.json();
  const cookieStore = await cookies();
  const sessionToken = cookieStore.get('session_token')?.value ?? crypto.randomUUID();
  try {
    const response = await fetch(`${GRADING_SERVICE_URL}/submissions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Session-Token': sessionToken },
      body: JSON.stringify({ ...payload, mode: 'submit', idempotencyKey: crypto.randomUUID() }),
    });
    const result = NextResponse.json(await response.json(), { status: response.status });
    result.cookies.set('session_token', sessionToken, { httpOnly: true, maxAge: 60 * 60 * 24 * 30, sameSite: 'lax' });
    return result;
  } catch {
    return NextResponse.json({ code: 'grading_unreachable', message: 'Grading service unreachable' }, { status: 502 });
  }
}
