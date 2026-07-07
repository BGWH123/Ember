import { NextResponse } from 'next/server';
import { GRADING_SERVICE_URL } from '@/lib/constants';
import { SubmissionResult } from '@/lib/types';

export async function POST(request: Request) {
  const { taskId, code, testIndices } = await request.json();

  try {
    const res = await fetch(`${GRADING_SERVICE_URL}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ taskId, code, testIndices }),
    });
    if (!res.ok) {
      const errText = await res.text();
      return NextResponse.json(
        { passed: 0, total: 0, allPassed: false, results: [], totalTimeMs: 0, error: errText },
        { status: 502 }
      );
    }
    const result: SubmissionResult = await res.json();
    return NextResponse.json(result);
  } catch {
    return NextResponse.json(
      { passed: 0, total: 0, allPassed: false, results: [], totalTimeMs: 0, error: 'Grading service unreachable' },
      { status: 502 }
    );
  }
}
