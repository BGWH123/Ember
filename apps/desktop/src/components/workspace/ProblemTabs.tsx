import { useEffect, useMemo, useState } from 'react';
import { FileCode2, Plus, X } from 'lucide-react';
import { useRouter } from '@/routing/navigation';
import { useLocale } from '@/context/LocaleContext';
import { cn } from '@/lib/utils';
import type { Problem } from '@/lib/types';

const STORAGE_KEY = 'ember:open-problems';
const MAX_OPEN_PROBLEMS = 12;

function loadOpenProblems(): string[] {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    return Array.isArray(value)
      ? value.filter((item): item is string => typeof item === 'string').slice(-MAX_OPEN_PROBLEMS)
      : [];
  } catch {
    return [];
  }
}

interface ProblemTabsProps {
  currentId: string;
  problems: Problem[];
  onAdd: () => void;
}

export function ProblemTabs({ currentId, problems, onAdd }: ProblemTabsProps) {
  const router = useRouter();
  const { locale } = useLocale();
  const [openIds, setOpenIds] = useState<string[]>(loadOpenProblems);
  const problemById = useMemo(
    () => new Map(problems.map((problem) => [problem.id, problem])),
    [problems],
  );

  useEffect(() => {
    setOpenIds((previous) => {
      if (previous.includes(currentId)) return previous;
      const next = [...previous, currentId].slice(-MAX_OPEN_PROBLEMS);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  }, [currentId]);

  function closeProblem(problemId: string) {
    const index = openIds.indexOf(problemId);
    const remaining = openIds.filter((id) => id !== problemId);
    setOpenIds(remaining);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(remaining));
    if (problemId !== currentId) return;
    const fallback = remaining[Math.min(index, remaining.length - 1)];
    router.push(fallback ? `/problems/${fallback}` : '/problems');
  }

  return (
    <div
      className="flex h-10 shrink-0 items-stretch overflow-hidden"
      style={{ background: 'var(--bg-sunken)', borderBottom: '1px solid var(--line)' }}
    >
      <div role="tablist" aria-label={locale === 'zh' ? '已打开的编程题' : 'Open coding problems'} className="flex min-w-0 flex-1 overflow-x-auto">
        {openIds.map((problemId) => {
          const problem = problemById.get(problemId);
          const title = problem
            ? (locale === 'zh' ? problem.titleZh : problem.title)
            : problemId;
          const active = problemId === currentId;
          return (
            <div
              key={problemId}
              className={cn(
                'group flex min-w-[150px] max-w-[230px] items-center border-r',
                active ? 'text-text' : 'text-text-3 hover:text-text-2',
              )}
              style={{
                borderColor: 'var(--line)',
                background: active ? 'var(--bg)' : 'transparent',
                boxShadow: active ? 'inset 0 2px 0 var(--accent)' : undefined,
              }}
            >
              <button
                type="button"
                role="tab"
                aria-selected={active}
                aria-label={`${problemId}.py`}
                title={title}
                onClick={() => router.push(`/problems/${problemId}`)}
                className="flex h-full min-w-0 flex-1 items-center gap-2 px-3 text-left"
              >
                <FileCode2 className={cn('h-3.5 w-3.5 shrink-0', active && 'text-accent')} />
                <span className="truncate text-xs">{title}</span>
              </button>
              <button
                type="button"
                aria-label={locale === 'zh' ? `关闭 ${problemId}.py` : `Close ${problemId}.py`}
                title={locale === 'zh' ? '关闭' : 'Close'}
                onClick={() => closeProblem(problemId)}
                className="mr-1 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded text-text-3 opacity-60 hover:bg-[color-mix(in_oklab,var(--text)_8%,transparent)] hover:text-text group-hover:opacity-100"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          );
        })}
      </div>
      <button
        type="button"
        aria-label={locale === 'zh' ? '打开其他编程题' : 'Open another coding problem'}
        title={locale === 'zh' ? '打开其他题目' : 'Open another problem'}
        onClick={onAdd}
        className="inline-flex w-10 shrink-0 items-center justify-center text-text-3 hover:bg-[color-mix(in_oklab,var(--text)_5%,transparent)] hover:text-text"
        style={{ borderLeft: '1px solid var(--line)' }}
      >
        <Plus className="h-4 w-4" />
      </button>
    </div>
  );
}
