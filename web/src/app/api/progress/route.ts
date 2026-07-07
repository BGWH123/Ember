import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { GRADING_SERVICE_URL } from '@/lib/constants';

export async function GET() {
  const cookieStore = await cookies();
  const sessionToken = cookieStore.get('session_token')?.value;

  if (!sessionToken) {
    return NextResponse.json({ progress: {} });
  }

  const userRes = await fetch(`${GRADING_SERVICE_URL}/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionToken }),
  });
  if (!userRes.ok) return NextResponse.json({ progress: {} });

  const { userId } = await userRes.json();
  const progressRes = await fetch(`${GRADING_SERVICE_URL}/progress/${userId}`);
  if (!progressRes.ok) return NextResponse.json({ progress: {} });

  const progress = await progressRes.json();
  return NextResponse.json({ progress });
}
