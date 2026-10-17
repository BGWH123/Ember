import { appFetch } from '@/transport/client';
import {useEffect,useState} from 'react';
import type { Problem } from '@/lib/types';

interface SolutionCell {
  type: string;
  source: string;
  role: string;
}

export function solutionCode(cells:SolutionCell[]):string {
  return cells.filter(cell=>cell.role==='solution').map(cell=>cell.source).join('\n\n').trim();
}
export async function getSolutionCode(problemId:string):Promise<string>{
  const response=await appFetch(`/api/solutions/${problemId}`);
  if(!response.ok) throw new Error('Could not load reference solution');
  return solutionCode((await response.json()).cells??[]);
}
export function useSolution(problemId:string){
  const [cells,setCells]=useState<SolutionCell[]>([]);
  useEffect(()=>{let alive=true;setCells([]);appFetch(`/api/solutions/${problemId}`).then(r=>r.json()).then(data=>{if(alive)setCells(data.cells??[]);});return()=>{alive=false;};},[problemId]);
  return cells;
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
