'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import {
  Sigma,
  ScatterChart,
  Brain,
  Layers,
  Sparkles,
  Cpu,
  Target,
  ChevronDown,
  ArrowRight,
  type LucideIcon,
} from 'lucide-react';
import { TopNav } from '@/components/layout/TopNav';
import { Footer } from '@/components/layout/Footer';
import { cn } from '@/lib/utils';

// ─── Types ───────────────────────────────────────────────────────────────

type ModuleStatus = 'hands-on' | 'theory' | 'planned';

interface CurriculumModule {
  id: string;
  title: string;
  status: ModuleStatus;
  blurb: string;
  problems: string[];
  hotCategories: string[];
  baguChapters: string[];
}

interface Stage {
  id: string;
  order: number;
  title: string;
  subtitle: string;
  icon: string;
  summary: string;
  modules: CurriculumModule[];
}

interface Curriculum {
  meta: {
    title: string;
    subtitle: string;
    description: string;
    legend: { key: string; label: string; desc: string }[];
  };
  stages: Stage[];
}

interface ProblemMeta {
  id: string;
  title: string;
  titleZh: string;
  difficulty: string;
}

// ─── Static maps ───────────────────────────────────────────────────────────

const ICONS: Record<string, LucideIcon> = {
  Sigma,
  ScatterChart,
  Brain,
  Layers,
  Sparkles,
  Cpu,
  Target,
};

const STATUS_STYLE: Record<ModuleStatus, { label: string; color: string }> = {
  'hands-on': { label: '动手', color: 'var(--easy)' },
  theory: { label: '专题', color: 'var(--accent)' },
  planned: { label: '规划中', color: 'var(--text-3)' },
};

const HOT_CAT_NAMES: Record<string, string> = {
  math: '数学与概率统计',
  ml: '经典机器学习',
  dl: '深度学习基础',
  transformer: 'Transformer 与注意力',
  llm: '大模型 LLM',
  train: '训练与微调',
  infer: '推理与部署优化',
  'rag-agent': 'RAG 与 Agent',
  align: '对齐与 RLHF',
};

// ─── Data hooks ──────────────────────────────────────────────────────────

function useCurriculum() {
  const [data, setData] = useState<Curriculum | null>(null);
  useEffect(() => {
    fetch('/curriculum.json')
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData(null));
  }, []);
  return data;
}

function useProblemMap() {
  const [map, setMap] = useState<Map<string, ProblemMeta>>(new Map());
  useEffect(() => {
    fetch('/problems-list.json')
      .then((r) => r.json())
      .then((d) => {
        const m = new Map<string, ProblemMeta>();
        (d.problems || []).forEach((p: ProblemMeta) => m.set(p.id, p));
        setMap(m);
      })
      .catch(() => setMap(new Map()));
  }, []);
  return map;
}

// ─── Module row ────────────────────────────────────────────────────────────

function ModuleRow({ mod, problemMap }: { mod: CurriculumModule; problemMap: Map<string, ProblemMeta> }) {
  const [open, setOpen] = useState(false);
  const status = STATUS_STYLE[mod.status];
  const hasLinks = mod.problems.length > 0 || mod.hotCategories.length > 0 || mod.baguChapters.length > 0;

  return (
    <div className="card overflow-hidden">
      <button
        onClick={() => hasLinks && setOpen((v) => !v)}
        className={cn(
          'flex w-full items-start gap-3.5 px-4 py-3.5 text-left transition-colors duration-150',
          hasLinks && 'hover:bg-[color-mix(in_oklab,var(--accent)_3%,var(--bg-elev))]',
        )}
      >
        <span
          className="mono mt-0.5 shrink-0 rounded-[5px] px-1.5 py-0.5 text-[10.5px] font-medium"
          style={{
            color: status.color,
            border: `1px solid color-mix(in oklab, ${status.color} 30%, var(--line))`,
            background: `color-mix(in oklab, ${status.color} 8%, var(--bg-elev))`,
          }}
        >
          {status.label}
        </span>
        <span className="flex-1">
          <span className="block text-[14.5px] font-semibold tracking-[-0.01em] text-text">{mod.title}</span>
          <span className="mt-0.5 block text-[13px] leading-relaxed text-text-2">{mod.blurb}</span>
        </span>
        {hasLinks && (
          <span className="mt-0.5 flex shrink-0 items-center gap-2">
            {mod.problems.length > 0 && (
              <span className="mono text-[11px] tabular-nums text-text-3">{mod.problems.length} 题</span>
            )}
            <ChevronDown className={cn('h-4 w-4 text-text-3 transition-transform duration-200', open && 'rotate-180')} />
          </span>
        )}
      </button>

      {open && hasLinks && (
        <div className="px-4 pb-4" style={{ borderTop: '1px solid var(--line)' }}>
          {mod.problems.length > 0 && (
            <div className="pt-3.5">
              <div className="eyebrow mb-2">编程题</div>
              <div className="flex flex-wrap gap-1.5">
                {mod.problems.map((pid) => {
                  const p = problemMap.get(pid);
                  return (
                    <Link
                      key={pid}
                      href={`/problems/${pid}`}
                      className="group inline-flex items-center gap-1.5 rounded-[7px] px-2.5 py-1 text-[12.5px] text-text-2 transition-colors hover:text-accent"
                      style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}
                    >
                      {p ? p.titleZh || p.title : pid}
                      <ArrowRight className="h-3 w-3 opacity-0 transition-opacity group-hover:opacity-100" />
                    </Link>
                  );
                })}
              </div>
            </div>
          )}

          {mod.hotCategories.length > 0 && (
            <div className="pt-3.5">
              <div className="eyebrow mb-2">热题解析</div>
              <div className="flex flex-wrap gap-1.5">
                {mod.hotCategories.map((c) => (
                  <Link
                    key={c}
                    href="/hot"
                    className="inline-flex items-center gap-1.5 rounded-[7px] px-2.5 py-1 text-[12.5px] text-text-2 transition-colors hover:text-accent"
                    style={{ background: 'var(--accent-wash)', border: '1px solid var(--accent-line)' }}
                  >
                    {HOT_CAT_NAMES[c] || c}
                  </Link>
                ))}
              </div>
            </div>
          )}

          {mod.baguChapters.length > 0 && (
            <div className="pt-3.5">
              <div className="eyebrow mb-2">八股专题</div>
              <div className="flex flex-wrap gap-1.5">
                {mod.baguChapters.map((ch) => (
                  <Link
                    key={ch}
                    href="/bagu"
                    className="inline-flex items-center rounded-[7px] px-2.5 py-1 text-[12.5px] text-text-2 transition-colors hover:text-accent"
                    style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}
                  >
                    {ch.replace(/_/g, ' ').replace(/大模型 LLMs /, '')}
                  </Link>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────

export default function CurriculumPage() {
  const data = useCurriculum();
  const problemMap = useProblemMap();

  const stats = useMemo(() => {
    if (!data) return { modules: 0, handsOn: 0 };
    let modules = 0;
    let handsOn = 0;
    data.stages.forEach((s) =>
      s.modules.forEach((m) => {
        modules++;
        if (m.status === 'hands-on') handsOn++;
      }),
    );
    return { modules, handsOn };
  }, [data]);

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
        <div className="mb-10 max-w-[820px] border-b pb-8" style={{ borderColor: 'var(--line)' }}>
          <div className="eyebrow mb-2.5">知识体系 · Curriculum</div>
          <h1 className="mb-3.5 text-[clamp(30px,4vw,46px)] font-semibold leading-[1.08] tracking-[-0.03em]">
            {data.meta.title}
          </h1>
          <p className="max-w-[64ch] text-[15px] leading-relaxed text-text-2">{data.meta.description}</p>
          <div className="mt-5 flex flex-wrap items-center gap-6 divider-dashed pt-5">
            {[
              { k: '学习阶段', v: String(data.stages.length) },
              { k: '知识模块', v: String(stats.modules) },
              { k: '动手模块', v: String(stats.handsOn) },
            ].map((m) => (
              <div key={m.k}>
                <div className="eyebrow">{m.k}</div>
                <div className="mono mt-1 text-sm tabular-nums">{m.v}</div>
              </div>
            ))}
            <div className="ml-auto flex flex-wrap gap-3">
              {data.meta.legend.map((l) => {
                const color = STATUS_STYLE[l.key as ModuleStatus]?.color || 'var(--text-3)';
                return (
                  <span key={l.key} className="inline-flex items-center gap-1.5 text-[12px] text-text-2" title={l.desc}>
                    <span className="h-2 w-2 rounded-full" style={{ background: color }} />
                    {l.label}
                  </span>
                );
              })}
            </div>
          </div>
        </div>

        {/* Stages */}
        <div className="space-y-12">
          {data.stages.map((stage) => {
            const Icon = ICONS[stage.icon] || Layers;
            return (
              <section key={stage.id}>
                <div className="mb-4 flex items-start gap-4">
                  <span
                    className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl text-accent"
                    style={{ border: '1px solid var(--accent-line)', background: 'var(--accent-wash)' }}
                  >
                    <Icon className="h-5 w-5" strokeWidth={1.6} />
                  </span>
                  <div className="flex-1">
                    <h2 className="text-[20px] font-semibold tracking-[-0.02em]">{stage.title}</h2>
                    <p className="mono mt-0.5 text-[12px] tracking-[0.04em] text-text-3">{stage.subtitle}</p>
                    <p className="mt-2 max-w-[72ch] text-[13.5px] leading-relaxed text-text-2">{stage.summary}</p>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2.5 max-[760px]:grid-cols-1">
                  {stage.modules.map((mod) => (
                    <ModuleRow key={mod.id} mod={mod} problemMap={problemMap} />
                  ))}
                </div>
              </section>
            );
          })}
        </div>
      </main>
      <Footer />
    </div>
  );
}
