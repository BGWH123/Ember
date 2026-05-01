import { NextResponse } from 'next/server';
import solutions from '@/lib/solutions.json';

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const data = (solutions as Record<string, { cells: unknown[] }>)[id];
  if (!data) {
    return NextResponse.json({ error: 'Solution not found' }, { status: 404 });
  }
  return NextResponse.json(data);
}
