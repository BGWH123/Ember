import solutionsData from '@/lib/solutions.json';
import type { Problem } from '@/lib/types';

interface SolutionCell {
  type: string;
  source: string;
  role: string;
}

export function getSolutionCode(problemId: string): string {
  const data = (solutionsData as Record<string, { cells: SolutionCell[] }>)[problemId];
  const cells = data?.cells ?? [];

  return cells
    .filter((cell) => cell.role === 'solution')
    .map((cell) => cell.source)
    .join('\n\n')
    .trim();
}

export function formatTestCode(code: string, functionName: string): string {
  return code
    .replace(/\{fn\}/g, functionName)
    .split('\n')
    .filter((line) => !line.startsWith('import ') && !line.startsWith('from '))
    .join('\n')
    .trim();
}

export function getSampleTests(problem: Problem, limit = 2): Array<{ name: string; code: string }> {
  return problem.tests.slice(0, limit).map((test) => ({
    name: test.name,
    code: formatTestCode(test.code, problem.functionName),
  }));
}
