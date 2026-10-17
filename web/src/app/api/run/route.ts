import { submitGrading } from '@/lib/grading-proxy';

export async function POST(request: Request) {
  return submitGrading(request, 'sample');
}
