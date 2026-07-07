'use client';

import { useEffect, useMemo, useState } from 'react';
import { TopNav } from '@/components/layout/TopNav';
import { Footer } from '@/components/layout/Footer';
import { MarkdownContent } from '@/components/workspace/MarkdownContent';
import { Badge } from '@/components/ui/Badge';
import { Search, ChevronDown, ExternalLink, Flame, ListFilter } from 'lucide-react';
import { cn } from '@/lib/utils';

// ─── Types ───────────────────────────────────────────────────────────────

interface Reference {
  label: string;
  url: string;
}

interface HotQuestion {
  id: string;
  category: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  tags: string[];
  question: string;
  answer: string;
  references: Reference[];
}

interface Category {
  id: string;
  name: string;
  order: number;
}

interface HotData {
  meta: { title: string; description: string; version: number };
  categories: Category[];
  questions: HotQuestion[];
}

// ─── Data hook ─────────────────────────────────────────────────────────────

function useHotData() {
  const [data, setData] = useState<HotData | null>(null);

  useEffect(() => {
    fetch('/hot-questions.json')
      .then((r) => r.json())
      .then((d) => setData(d))
      .catch(() => setData(null));
  }, []);

  return data;
}

// ─── Question card ─────────────────────────────────────────────────────────

function QuestionCard({
  index,
  q,
  categoryName,
}: {
  index: number;
  q: HotQuestion;
  categoryName: string;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="card overflow-hidden">
      {/* Question header — click to toggle the answer */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start gap-4 px-5 py-4 text-left transition-colors duration-150 hover:bg-[color-mix(in_oklab,var(--accent)_3%,var(--bg-elev))]"
      >
        <span className="mono mt-0.5 shrink-0 text-[13px] tabular-nums text-text-3">
          {String(index).padStart(2, '0')}
        </span>
        <span className="flex-1 text-[15px] font-semibold leading-relaxed tracking-[-0.01em] text-text">
          {q.question}
        </span>
        <span className="flex shrink-0 items-center gap-2.5">
          <Badge variant={q.difficulty.toLowerCase() as 'easy' | 'medium' | 'hard'}>
            {q.difficulty.toUpperCase()}
          </Badge>
          <ChevronDown
            className={cn('h-4 w-4 text-text-3 transition-transform duration-200', open && 'rotate-180')}
          />
        </span>
      </button>

      {/* Answer body */}
      {open && (
        <div className="px-5 pb-5" style={{ borderTop: '1px solid var(--line)' }}>
          <div className="flex flex-wrap items-center gap-2 py-3.5">
            <span className="eyebrow">{categoryName}</span>
            {q.tags.map((tag) => (
              <span
                key={tag}
                className="mono rounded-[5px] px-1.5 py-0.5 text-[11px] text-text-3"
                style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}
              >
                {tag}
              </span>
            ))}
          </div>

          <MarkdownContent content={q.answer} />

          {q.references.length > 0 && (
            <div className="mt-5 pt-4" style={{ borderTop: '1px dashed var(--line)' }}>
              <div className="eyebrow mb-2.5">参考来源 · References</div>
              <ul className="space-y-1.5">
                {q.references.map((ref) => (
                  <li key={ref.url}>
                    <a
                      href={ref.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="group inline-flex items-start gap-1.5 text-[13px] leading-relaxed text-text-2 transition-colors hover:text-accent"
                    >
                      <ExternalLink className="mt-[3px] h-3 w-3 shrink-0 text-text-3 transition-colors group-hover:text-accent" />
                      <span>{ref.label}</span>
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────

export default function HotQuestionsPage() {
  const data = useHotData();
  const [search, setSearch] = useState('');
  const [activeCat, setActiveCat] = useState<string>('');

  const categories = useMemo(
    () => (data ? [...data.categories].sort((a, b) => a.order - b.order) : []),
    [data],
  );
  const catName = useMemo(() => {
    const m = new Map<string, string>();
    categories.forEach((c) => m.set(c.id, c.name));
    return m;
  }, [categories]);

  const counts = useMemo(() => {
    const m = new Map<string, number>();
    data?.questions.forEach((q) => m.set(q.category, (m.get(q.category) || 0) + 1));
    return m;
  }, [data]);

  const filtered = useMemo(() => {
    if (!data) return [];
    const s = search.trim().toLowerCase();
    return data.questions.filter((q) => {
      if (activeCat && q.category !== activeCat) return false;
      if (s) {
        const hay = (q.question + ' ' + q.answer + ' ' + q.tags.join(' ')).toLowerCase();
        if (!hay.includes(s)) return false;
      }
      return true;
    });
  }, [data, search, activeCat]);

  if (!data) {
    return (
      <div className="min-h-screen bg-bg">
        <TopNav />
        <div className="flex items-center justify-center py-32">
          <p className="text-sm text-text-3">加载中…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-bg">
      <TopNav />
      <main className="mx-auto max-w-[1100px] px-7 pb-20 pt-10">
        {/* Header */}
        <div className="mb-9 max-w-[760px] border-b pb-7" style={{ borderColor: 'var(--line)' }}>
          <div className="eyebrow mb-2.5 flex items-center gap-2">
            <Flame className="h-3.5 w-3.5 text-accent" />
            热题 50 · Hot 50
          </div>
          <h1 className="mb-3.5 text-[clamp(30px,4vw,46px)] font-semibold leading-[1.08] tracking-[-0.03em]">
            {data.meta.title}
          </h1>
          <p className="max-w-[60ch] text-[15px] leading-relaxed text-text-2">{data.meta.description}</p>
          <div className="mt-5 flex flex-wrap gap-6 divider-dashed pt-5">
            {[
              { k: '题目数', v: String(data.questions.length) },
              { k: '知识板块', v: String(categories.length) },
              { k: '标注来源', v: `${data.questions.reduce((a, q) => a + q.references.length, 0)} 处` },
            ].map((m) => (
              <div key={m.k}>
                <div className="eyebrow">{m.k}</div>
                <div className="mono mt-1 text-sm tabular-nums">{m.v}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-[220px_minmax(0,1fr)] items-start gap-10 max-[860px]:grid-cols-1">
          {/* Sidebar: categories */}
          <aside className="sticky top-[76px] max-[860px]:static">
            <h5 className="eyebrow mb-2.5 flex items-center gap-1.5">
              <ListFilter className="h-3 w-3" /> 板块
            </h5>
            <div className="flex flex-col gap-px">
              <button
                onClick={() => setActiveCat('')}
                className={cn(
                  'flex items-center justify-between gap-2.5 rounded-[7px] px-2.5 py-[7px] text-[13.5px] transition-[background,color] duration-150',
                  !activeCat
                    ? 'text-accent'
                    : 'text-text-2 hover:bg-[color-mix(in_oklab,var(--text)_5%,transparent)] hover:text-text',
                )}
                style={!activeCat ? { background: 'var(--accent-wash)' } : undefined}
              >
                <span>全部</span>
                <span className="mono text-[11.5px] tabular-nums text-text-3">{data.questions.length}</span>
              </button>
              {categories.map((c) => (
                <button
                  key={c.id}
                  onClick={() => setActiveCat(c.id === activeCat ? '' : c.id)}
                  className={cn(
                    'flex items-center justify-between gap-2.5 rounded-[7px] px-2.5 py-[7px] text-left text-[13.5px] transition-[background,color] duration-150',
                    activeCat === c.id
                      ? 'text-accent'
                      : 'text-text-2 hover:bg-[color-mix(in_oklab,var(--text)_5%,transparent)] hover:text-text',
                  )}
                  style={activeCat === c.id ? { background: 'var(--accent-wash)' } : undefined}
                >
                  <span className="truncate">{c.name}</span>
                  <span
                    className={cn(
                      'mono text-[11.5px] tabular-nums',
                      activeCat === c.id ? 'text-accent opacity-80' : 'text-text-3',
                    )}
                  >
                    {counts.get(c.id) || 0}
                  </span>
                </button>
              ))}
            </div>
          </aside>

          {/* Question list */}
          <div>
            {/* Search */}
            <div className="field mb-4 flex h-[38px] items-center gap-2 px-3">
              <Search className="h-3.5 w-3.5 shrink-0 text-text-3" />
              <input
                type="text"
                placeholder="搜索题目、关键词、标签…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="flex-1 bg-transparent text-sm text-text outline-none placeholder:text-text-3"
              />
              {search && (
                <button onClick={() => setSearch('')} className="text-xs text-text-3 hover:text-text">
                  清除
                </button>
              )}
            </div>

            <div className="mb-3 mono text-[11.5px] text-text-3">
              共 {filtered.length} 题{activeCat ? ` · ${catName.get(activeCat)}` : ''}
            </div>

            <div className="space-y-2.5">
              {filtered.map((q, i) => (
                <QuestionCard key={q.id} index={i + 1} q={q} categoryName={catName.get(q.category) || ''} />
              ))}
            </div>

            {filtered.length === 0 && (
              <p className="py-16 text-center text-sm text-text-3">没有匹配的题目。</p>
            )}
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}
