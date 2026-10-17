import { useEffect, useState, type CSSProperties } from 'react';
import { useLocale } from '@/context/LocaleContext';

/** Nine tensor cells carry a quiet wave across a pair of code brackets. */
export function EmberActivity({ small = false }: { small?: boolean }) {
  return <span className={`ember-activity${small ? ' ember-activity--small' : ''}`} aria-hidden="true">
    <span className="ember-activity__bracket" />
    <span className="ember-activity__matrix">
      {Array.from({ length: 9 }, (_, i) => <span key={i} style={{ '--cell': (i % 3) + Math.floor(i / 3) } as CSSProperties} />)}
    </span>
    <span className="ember-activity__bracket ember-activity__bracket--end" />
  </span>;
}

export function WaitingState({ label, description, compact = false, evaluation = false }: {
  label?: string; description?: string; compact?: boolean; evaluation?: boolean;
}) {
  const { locale } = useLocale();
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!evaluation) return;
    const started = Date.now();
    const timer = window.setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [evaluation]);
  const zh = locale === 'zh';
  return <div className={`waiting-state${compact ? ' waiting-state--compact' : ''}`} role="status" aria-live="polite" aria-atomic="true">
    <div className="waiting-state__art"><EmberActivity /></div>
    <div className="waiting-state__copy">
      <p className="waiting-state__label">{label ?? (zh ? '准备你的工作台' : 'Preparing your workspace')}</p>
      <p className="waiting-state__description">{description ?? (evaluation
        ? (zh ? '代码正在运行，结果就绪后会自动显示。' : 'Your code is running. Results will appear here.')
        : (zh ? '正在整理题目与学习记录，马上就好。' : 'Gathering your exercises and learning progress.'))}</p>
      {evaluation && <span className="waiting-state__elapsed" aria-hidden="true">{elapsed}s{elapsed >= 8 ? (zh ? ' · 首次运行需要加载评测环境' : ' · First runs may need to load the evaluator') : ''}</span>}
    </div>
  </div>;
}
