import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { GRADING_SERVICE_URL } from '@/lib/constants';

type ProxyOptions = {
  started: number;
  sessionToken: string;
  method?: 'GET' | 'POST';
  payload?: unknown;
  transform?: (body: unknown) => unknown;
  setSession?: boolean;
};

/** Preserve status/idempotency and expose the backend and proxy spans together. */
export async function proxyGrading(request: Request, path: string, options: ProxyOptions) {
  const supplied = request.headers.get('X-Request-ID') ?? '';
  const requestId = /^[A-Za-z0-9_-]{1,64}$/.test(supplied) ? supplied : crypto.randomUUID();
  const upstreamStarted = performance.now();
  try {
    const response = await fetch(`${GRADING_SERVICE_URL}${path}`, {
      method: options.method ?? 'GET',
      headers: { 'Content-Type': 'application/json', 'X-Session-Token': options.sessionToken, 'X-Request-ID': requestId },
      body: options.payload === undefined ? undefined : JSON.stringify(options.payload),
      cache: 'no-store',
      signal: AbortSignal.timeout(5000),
    });
    const body = await response.json();
    const upstreamMs = performance.now() - upstreamStarted;
    const result = NextResponse.json(response.ok && options.transform ? options.transform(body) : body, { status: response.status });
    if (options.setSession) {
      result.cookies.set('session_token', options.sessionToken, { httpOnly: true, maxAge: 60 * 60 * 24 * 30, sameSite: 'lax' });
    }
    const webMs = performance.now() - options.started;
    const spans = response.headers.get('Server-Timing');
    result.headers.set('X-Request-ID', response.headers.get('X-Request-ID') ?? requestId);
    result.headers.set('Server-Timing', [spans,
      `proxy_upstream;dur=${upstreamMs.toFixed(3)}`,
      `proxy_overhead;dur=${Math.max(0, webMs - upstreamMs).toFixed(3)}`,
      `web;dur=${webMs.toFixed(3)}`,
    ].filter(Boolean).join(', '));
    return result;
  } catch (error) {
    const timeout = error instanceof Error && (error.name === 'TimeoutError' || error.name === 'AbortError');
    return NextResponse.json({ code: timeout ? 'grading_timeout' : 'grading_unreachable', message: timeout ? 'Grading service timed out' : 'Grading service unreachable' }, {
      status: timeout ? 504 : 502,
      headers: { 'X-Request-ID': requestId, 'Server-Timing': `web;dur=${(performance.now() - options.started).toFixed(3)}` },
    });
  }
}

export async function submitGrading(request: Request, mode: 'sample' | 'submit') {
  const started = performance.now();
  let payload;
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({ code: 'invalid_json', message: 'Invalid JSON body' }, { status: 400 });
  }
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return NextResponse.json({ code: 'invalid_payload', message: 'Expected a submission object' }, { status: 422 });
  }
  const sessionToken = (await cookies()).get('session_token')?.value ?? crypto.randomUUID();
  return proxyGrading(request, '/submissions', {
    started, sessionToken, method: 'POST', setSession: true,
    payload: { ...payload, mode, idempotencyKey: payload.idempotencyKey ?? crypto.randomUUID() },
  });
}
