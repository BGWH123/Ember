'use client';

import { Search } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useLocale } from '@/context/LocaleContext';

interface ProblemFiltersProps {
  search: string;
  onSearchChange: (v: string) => void;
  difficulty: string;
  onDifficultyChange: (v: string) => void;
  status: string;
  onStatusChange: (v: string) => void;
}

const difficulties = ['All', 'Easy', 'Medium', 'Hard'] as const;
const statuses = ['All', 'Todo', 'Attempted', 'Solved'] as const;

export function ProblemFilters({
  search, onSearchChange,
  difficulty, onDifficultyChange,
  status, onStatusChange,
}: ProblemFiltersProps) {
  const { t } = useLocale();
  return (
    <div className="flex items-center gap-4 mb-6">
      <div className="relative flex-1 max-w-sm">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-tertiary" />
        <input
          type="text"
          placeholder={t('searchPlaceholder')}
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
          className="w-full h-9 pl-9 pr-3 rounded-lg bg-surface-secondary border border-border text-sm text-text-primary placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition-colors"
        />
      </div>
      <div className="flex items-center gap-1">
        {difficulties.map((d) => (
          <button
            key={d}
            onClick={() => onDifficultyChange(d === 'All' ? '' : d)}
            className={cn(
              'px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
              (d === 'All' && !difficulty) || difficulty === d
                ? 'bg-accent/10 text-accent'
                : 'text-text-secondary hover:bg-gray-100'
            )}
          >
            {t(d)}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-1">
        {statuses.map((s) => (
          <button
            key={s}
            onClick={() => onStatusChange(s === 'All' ? '' : s.toLowerCase())}
            className={cn(
              'px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
              (s === 'All' && !status) || status === s.toLowerCase()
                ? 'bg-accent/10 text-accent'
                : 'text-text-secondary hover:bg-gray-100'
            )}
          >
            {t(s)}
          </button>
        ))}
      </div>
    </div>
  );
}
