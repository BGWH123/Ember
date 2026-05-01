'use client';

import Link from 'next/link';
import { Layers, Eye, Brain, Zap, Target, Microscope, Sparkles, Wand2 } from 'lucide-react';
import { useLocale } from '@/context/LocaleContext';
import type { LearningPath } from '@/lib/types';

const ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  Layers,
  Eye,
  Brain,
  Zap,
  Target,
  Microscope,
  Sparkles,
  Wand2,
};

interface PathCardProps {
  path: LearningPath & { solved: number; total: number };
}

export function PathCard({ path }: PathCardProps) {
  const { locale, t } = useLocale();
  const title = locale === 'zh' ? path.titleZh : path.titleEn;
  const description = locale === 'zh' ? path.descriptionZh : path.descriptionEn;
  const Icon = ICONS[path.icon] ?? Layers;
  const pct = path.total > 0 ? Math.round((path.solved / path.total) * 100) : 0;
  const done = path.solved === path.total;

  return (
    <Link
      href={`/paths/${path.id}`}
      className="group block rounded-2xl border border-border bg-white p-5 hover:border-accent/40 hover:shadow-sm transition-all duration-200"
    >
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-accent/8 flex items-center justify-center">
          <Icon className="w-5 h-5 text-accent" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="text-sm font-semibold text-text-primary truncate">{title}</h3>
            {done && (
              <span className="flex-shrink-0 text-xs font-medium text-green-600 bg-green-50 px-2 py-0.5 rounded-full">
                {t('completedBadge')}
              </span>
            )}
          </div>
          <p className="text-xs text-text-secondary line-clamp-2 mb-3">{description}</p>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-accent rounded-full transition-all duration-300"
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="text-xs text-text-tertiary flex-shrink-0">
              {t('pathProgress', { solved: path.solved, total: path.total })}
            </span>
          </div>
        </div>
      </div>
    </Link>
  );
}
