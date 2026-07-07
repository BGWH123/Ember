'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  ArrowRight,
  Check,
  FlaskConical,
  BarChart3,
  Library,
  Sigma,
  ScatterChart,
  Brain,
  Layers,
  Sparkles,
  Cpu,
  Target,
  type LucideIcon,
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { useLocale } from '@/context/LocaleContext';
import type { LearningPath } from '@/lib/types';

interface HomeContentProps {
  stats: { total: number; easy: number; medium: number; hard: number };
}

type PathWithProgress = LearningPath & { solved: number; total: number };

type StageStatus = 'hands-on' | 'theory' | 'planned';

interface CurriculumStage {
  id: string;
  order: number;
  title: string;
  subtitle: string;
  icon: string;
  summary: string;
  modules: { id: string; title: string; status: StageStatus; blurb: string }[];
}

interface Curriculum {
  meta: { title: string; subtitle: string; description: string; legend: { key: string; label: string; desc: string }[] };
  stages: CurriculumStage[];
}

const STAGE_ICONS: Record<string, LucideIcon> = {
  Sigma,
  ScatterChart,
  Brain,
  Layers,
  Sparkles,
  Cpu,
  Target,
};

function SectionHeader({
  eyebrow,
  title,
  linkText,
  linkHref,
}: {
  eyebrow: string;
  title: string;
  linkText?: string;
  linkHref?: string;
}) {
  return (
    <div className="mb-8 flex items-baseline justify-between gap-6">
      <div>
        <div className="eyebrow mb-2">{eyebrow}</div>
        <h2 className="mt-2 text-[clamp(24px,2.6vw,32px)] font-semibold tracking-[-0.025em]">{title}</h2>
      </div>
      {linkText && linkHref && (
        <Link
          href={linkHref}
          className="mono flex shrink-0 items-center gap-1.5 text-[13px] text-text-2 transition-colors hover:text-text"
        >
          {linkText} →
        </Link>
      )}
    </div>
  );
}

const CODE_LINES = [
  { n: 1, code: '<span style="color:color-mix(in oklab,var(--accent) 80%,var(--text))"># Implement causal self-attention.</span>', style: 'italic' },
  { n: 2, code: '<span style="color:color-mix(in oklab,var(--accent) 80%,var(--text))">import</span> <span style="font-weight:500">torch</span>' },
  { n: 3, code: '<span style="color:color-mix(in oklab,var(--accent) 80%,var(--text))">import</span> <span style="font-weight:500">torch.nn.functional</span> <span style="color:color-mix(in oklab,var(--accent) 80%,var(--text))">as</span> <span style="font-weight:500">F</span>' },
  { n: 4, code: '' },
  { n: 5, code: '<span style="color:color-mix(in oklab,var(--accent) 80%,var(--text))">def</span> <span style="font-weight:500">causal_attention</span><span style="color:var(--text-2)">(</span>q<span style="color:var(--text-2)">,</span> k<span style="color:var(--text-2)">,</span> v<span style="color:var(--text-2)">):</span>' },
  { n: 6, code: '    d <span style="color:var(--text-2)">=</span> q<span style="color:var(--text-2)">.</span>size<span style="color:var(--text-2)">(-</span><span style="color:color-mix(in oklab,var(--hard) 65%,var(--text))">1</span><span style="color:var(--text-2)">)</span>' },
  { n: 7, code: '    scores <span style="color:var(--text-2)">=</span> q <span style="color:var(--text-2)">@</span> k<span style="color:var(--text-2)">.</span>transpose<span style="color:var(--text-2)">(-</span><span style="color:color-mix(in oklab,var(--hard) 65%,var(--text))">2</span><span style="color:var(--text-2)">,-</span><span style="color:color-mix(in oklab,var(--hard) 65%,var(--text))">1</span><span style="color:var(--text-2)">)</span> <span style="color:var(--text-2)">/</span> d<span style="color:var(--text-2)">**</span><span style="color:color-mix(in oklab,var(--hard) 65%,var(--text))">0.5</span>' },
  { n: 8, code: '    T <span style="color:var(--text-2)">=</span> q<span style="color:var(--text-2)">.</span>size<span style="color:var(--text-2)">(-</span><span style="color:color-mix(in oklab,var(--hard) 65%,var(--text))">2</span><span style="color:var(--text-2)">)</span>' },
  { n: 9, code: '    mask <span style="color:var(--text-2)">=</span> torch<span style="color:var(--text-2)">.</span>triu<span style="color:var(--text-2)">(</span>torch<span style="color:var(--text-2)">.</span>ones<span style="color:var(--text-2)">(</span>T<span style="color:var(--text-2)">,</span>T<span style="color:var(--text-2)">),</span> <span style="color:color-mix(in oklab,var(--hard) 65%,var(--text))">1</span><span style="color:var(--text-2)">).</span>bool<span style="color:var(--text-2)">()</span>' },
  { n: 10, code: '    scores <span style="color:var(--text-2)">=</span> scores<span style="color:var(--text-2)">.</span>masked_fill<span style="color:var(--text-2)">(</span>mask<span style="color:var(--text-2)">,</span> <span style="color:color-mix(in oklab,var(--accent) 80%,var(--text))">float</span><span style="color:var(--text-2)">(</span><span style="color:color-mix(in oklab,var(--easy) 70%,var(--text))">&#39;-inf&#39;</span><span style="color:var(--text-2)">))</span>' },
  { n: 11, code: '    <span style="color:color-mix(in oklab,var(--accent) 80%,var(--text))">return</span> F<span style="color:var(--text-2)">.</span>softmax<span style="color:var(--text-2)">(</span>scores<span style="color:var(--text-2)">,</span> <span style="color:var(--text-2)">-</span><span style="color:color-mix(in oklab,var(--hard) 65%,var(--text))">1</span><span style="color:var(--text-2)">)</span> <span style="color:var(--text-2)">@</span> v' },
];

const TESTS = [
  { name: 'shape_check', status: 'pass' as const, time: '3.2ms' },
  { name: 'masked_entries', status: 'pass' as const, time: '4.9ms' },
  { name: 'softmax_rows', status: 'pass' as const, time: '4.1ms' },
  { name: 'grad_flow', status: 'run' as const, time: 'running' },
];

export function HomeContent({ stats }: HomeContentProps) {
  const { locale, t } = useLocale();
  const [paths, setPaths] = useState<PathWithProgress[]>([]);
  const [curriculum, setCurriculum] = useState<Curriculum | null>(null);

  useEffect(() => {
    fetch('/api/paths')
      .then((r) => r.json())
      .then((d) => setPaths(d.paths ?? []))
      .catch(() => setPaths([]));
    fetch('/curriculum.json')
      .then((r) => r.json())
      .then(setCurriculum)
      .catch(() => setCurriculum(null));
  }, [locale]);

  return (
    <main className="mx-auto max-w-[1200px] px-7">
      {/* Hero */}
      <section className="pb-24 pt-20">
        <div className="grid grid-cols-1 items-center gap-16 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
          <div>
            <span
              className="mb-7 inline-flex items-center gap-2 rounded-pill px-2.5 py-[5px] text-xs text-text-2 mono"
              style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
            >
              <span className="font-semibold text-text">{stats.total}</span>
              <span>{locale === 'zh' ? '道动手编程题' : 'coding problems'}</span>
              <span className="text-text-3">·</span>
              <span className="text-text-3">no GPU required</span>
            </span>

            <h1 className="mb-5 text-[clamp(36px,5vw,56px)] font-semibold leading-[1.05] tracking-[-0.035em]">
              {locale === 'zh' ? (
                <>
                  从零实现
                  <br />
                  <span className="text-accent">现代 AI 系统</span>的内部
                </>
              ) : (
                <>
                  Implement the internals
                  <br />
                  of modern <span className="text-accent">AI systems</span>
                </>
              )}
            </h1>

            <p className="mb-8 max-w-[52ch] text-base leading-relaxed text-text-2">
              {locale === 'zh'
                ? '读论文,写代码。覆盖数学基础到大模型系统设计,在浏览器里动手实现 Transformer、注意力、RLHF、扩散与分布式训练的每一处细节。'
                : 'Read the paper, then write the code. From math foundations to LLM system design — implement every detail of Transformers, attention, RLHF, diffusion, and distributed training, right in your browser.'}
            </p>

            <div className="flex items-center gap-3">
              <Link href="/problems">
                <Button size="lg">
                  {t('startPracticing')}
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
              <Link href="/curriculum">
                <Button variant="secondary" size="lg">
                  {locale === 'zh' ? '查看知识体系' : 'Knowledge map'}
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
            </div>

            <div className="mt-8 flex gap-6 border-t pt-5" style={{ borderColor: 'var(--line)' }}>
              {[
                { k: t('metaTotal'), v: t('metaTotalVal', { n: stats.total }) },
                { k: t('metaCoverage'), v: locale === 'zh' ? '7 大阶段' : '7 stages' },
                { k: t('metaJudge'), v: 'torch_judge' },
              ].map((m) => (
                <div key={m.k}>
                  <div className="eyebrow">{m.k}</div>
                  <div className="mono mt-1 text-sm">{m.v}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Editor preview with line numbers + test sidebar */}
          <div
            className="relative hidden overflow-hidden rounded-[14px] lg:block"
            style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
          >
            <div
              className="pointer-events-none absolute inset-0 opacity-[0.35]"
              style={{
                backgroundImage: 'radial-gradient(var(--line) 1px, transparent 1px)',
                backgroundSize: '18px 18px',
                maskImage: 'radial-gradient(ellipse at 30% 30%, black 40%, transparent 75%)',
                WebkitMaskImage: 'radial-gradient(ellipse at 30% 30%, black 40%, transparent 75%)',
              }}
            />
            <div
              className="relative flex h-10 items-center gap-2.5 px-3.5 text-xs mono text-text-2"
              style={{ borderBottom: '1px solid var(--line)', background: 'color-mix(in oklab, var(--text) 2%, var(--bg-elev))' }}
            >
              <span className="flex gap-1.5">
                <span className="h-[9px] w-[9px] rounded-full" style={{ background: 'color-mix(in oklab, var(--hard) 60%, transparent)' }} />
                <span className="h-[9px] w-[9px] rounded-full" style={{ background: 'color-mix(in oklab, var(--medium) 60%, transparent)' }} />
                <span className="h-[9px] w-[9px] rounded-full" style={{ background: 'color-mix(in oklab, var(--easy) 60%, transparent)' }} />
              </span>
              <span>
                <span className="text-text-3">attention / </span>
                <span className="text-text">causal_self_attention.py</span>
              </span>
              <span
                className="ml-auto rounded-[6px] px-2 py-[3px] mono text-[11px]"
                style={{ border: '1px solid var(--line)', background: 'var(--bg-sunken)' }}
              >
                MEDIUM · 38%
              </span>
            </div>
            <div className="relative grid grid-cols-[1fr_280px] max-[720px]:grid-cols-1">
              <div className="overflow-hidden p-4 mono text-[12.5px] leading-[1.75]">
                {CODE_LINES.map((line) => (
                  <div key={line.n} className="grid grid-cols-[28px_1fr] gap-3.5">
                    <span className="select-none text-right text-text-3">{line.n}</span>
                    <span
                      className="whitespace-pre"
                      style={line.style === 'italic' ? { fontStyle: 'italic' } : undefined}
                      dangerouslySetInnerHTML={{ __html: line.code || '&nbsp;' }}
                    />
                  </div>
                ))}
              </div>
              <div
                className="flex max-[720px]:hidden flex-col gap-2 p-3.5"
                style={{ borderLeft: '1px solid var(--line)', background: 'var(--bg-sunken)' }}
              >
                <div className="py-0.5 pb-1.5 mono text-[11px] uppercase tracking-[0.1em] text-text-3">
                  Tests · {TESTS.length}
                </div>
                {TESTS.map((test) => (
                  <div
                    key={test.name}
                    className="flex items-center gap-2.5 rounded-lg px-2.5 py-2 mono text-xs"
                    style={{ background: 'var(--bg-elev)', border: '1px solid var(--line)' }}
                  >
                    <span
                      className="flex h-4 w-4 items-center justify-center rounded-[5px] text-[10px]"
                      style={
                        test.status === 'pass'
                          ? { background: 'var(--easy)', color: '#fff' }
                          : { background: 'transparent', color: 'var(--text-3)', border: '1px solid var(--line)' }
                      }
                    >
                      {test.status === 'pass' ? '✓' : '·'}
                    </span>
                    <span className="text-text">{test.name}</span>
                    <span className="ml-auto inline-flex items-center gap-1.5 text-text-3">
                      {test.time}
                      {test.status === 'run' && (
                        <span
                          className="h-1.5 w-1.5 animate-pulse rounded-full"
                          style={{ background: 'var(--accent)' }}
                        />
                      )}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Ticker */}
      <div
        className="overflow-hidden text-xs text-text-2 mono"
        style={{
          padding: '14px 0',
          borderTop: '1px solid var(--line)',
          borderBottom: '1px solid var(--line)',
          maskImage: 'linear-gradient(90deg, transparent, black 10%, black 90%, transparent)',
          WebkitMaskImage: 'linear-gradient(90deg, transparent, black 10%, black 90%, transparent)',
        }}
        aria-hidden="true"
      >
        <div className="flex w-max shrink-0 gap-10 animate-ticker">
          {[0, 1].map((copy) =>
            [
              'MultiHeadAttention',
              'Flash Attention (tiled)',
              'Rotary Position Embedding',
              'DPO Loss',
              'GRPO Loss',
              'Speculative Decoding',
              'Paged Attention',
              'LoRA / QLoRA',
              'Mamba SSM',
              'Mixture of Experts',
              'FSDP Training Step',
              'Ring Attention',
              'Flow Matching',
              'adaLN-Zero',
              'Multi-Token Prediction',
            ].map((name) => (
              <span key={`${copy}-${name}`} className="inline-flex shrink-0 items-center gap-2 whitespace-nowrap">
                <span className="h-1 w-1 rounded-full" style={{ background: 'var(--text-3)' }} />
                {name}
              </span>
            )),
          )}
        </div>
      </div>

      {/* Knowledge system */}
      {curriculum && (
        <section className="py-20" style={{ borderTop: '1px solid var(--line)' }}>
          <SectionHeader
            eyebrow={locale === 'zh' ? '§ 01 — 知识体系' : '§ 01 — Knowledge system'}
            title={locale === 'zh' ? '七阶段,由浅入深。' : 'Seven stages. From fundamentals to systems.'}
            linkText={locale === 'zh' ? '完整地图' : 'Full map'}
            linkHref="/curriculum"
          />
          <div className="grid grid-cols-2 gap-3.5 max-[760px]:grid-cols-1">
            {curriculum.stages.map((stage) => {
              const Icon = STAGE_ICONS[stage.icon] || Layers;
              return (
                <Link
                  key={stage.id}
                  href="/curriculum"
                  className="card card-hover group flex flex-col gap-3 p-5"
                >
                  <div className="flex items-center gap-3">
                    <span
                      className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-accent"
                      style={{ border: '1px solid var(--accent-line)', background: 'var(--accent-wash)' }}
                    >
                      <Icon className="h-4 w-4" strokeWidth={1.6} />
                    </span>
                    <div>
                      <h3 className="text-[15px] font-semibold tracking-[-0.012em]">{stage.title}</h3>
                      <p className="mono text-[11px] text-text-3">{stage.subtitle}</p>
                    </div>
                  </div>
                  <p className="text-[13px] leading-relaxed text-text-2">{stage.summary}</p>
                  <div className="mt-auto flex items-center gap-2 text-[11.5px] text-text-3 mono">
                    {stage.modules.length} 个模块
                    <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-[2px] group-hover:text-accent" />
                  </div>
                </Link>
              );
            })}
          </div>
        </section>
      )}

      {/* Learning paths */}
      {paths.length > 0 && (
        <section className="py-20" style={{ borderTop: '1px solid var(--line)' }}>
          <SectionHeader
            eyebrow={locale === 'zh' ? '§ 02 — 学习路径' : '§ 02 — Learning paths'}
            title={locale === 'zh' ? '选一条路线,循序渐进。' : 'Pick a track, go deep.'}
            linkText={locale === 'zh' ? `全部 ${paths.length} 条路径` : `All ${paths.length} paths`}
            linkHref="/paths"
          />
          <div className="grid grid-cols-3 gap-3.5 max-[900px]:grid-cols-2 max-[600px]:grid-cols-1">
            {paths.map((path, i) => {
              const title = locale === 'zh' ? path.titleZh : path.titleEn;
              const desc = locale === 'zh' ? path.descriptionZh : path.descriptionEn;
              const pct = path.total > 0 ? Math.round((path.solved / path.total) * 100) : 0;
              const tag = `PATH_${String(i + 1).padStart(2, '0')}`;
              return (
                <Link
                  key={path.id}
                  href={`/paths/${path.id}`}
                  className="card card-hover group flex flex-col gap-3.5 p-5"
                >
                  <span className="mono text-[10.5px] tracking-[0.12em] text-text-3">{tag}</span>
                  <h3 className="text-[15.5px] font-semibold tracking-[-0.012em]">{title}</h3>
                  <p className="text-[13px] leading-relaxed text-text-2">{desc}</p>
                  <div className="mt-auto flex items-center gap-2.5 mono text-[11.5px] text-text-2">
                    <span>
                      {Math.round((pct / 100) * path.total)}/{path.total}
                    </span>
                    <div className="h-[3px] flex-1 rounded-pill" style={{ background: 'var(--line)' }}>
                      <div className="h-full rounded-pill" style={{ width: `${pct}%`, background: 'var(--accent)' }} />
                    </div>
                    <span className="tabular-nums">{pct}%</span>
                  </div>
                </Link>
              );
            })}
          </div>
        </section>
      )}

      {/* Features */}
      <section className="py-20" style={{ borderTop: '1px solid var(--line)' }}>
        <SectionHeader
          eyebrow={locale === 'zh' ? '§ 03 — 工作方式' : '§ 03 — How it works'}
          title={locale === 'zh' ? '读论文,然后写代码。' : 'Read the paper, then write the code.'}
        />
        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          {[
            { num: '01', icon: Check, title: t('feat1Title'), desc: t('feat1Desc') },
            { num: '02', icon: FlaskConical, title: t('feat2Title'), desc: t('feat2Desc') },
            { num: '03', icon: BarChart3, title: t('feat3Title'), desc: t('feat3Desc') },
          ].map((f) => (
            <div key={f.num} className="pt-5" style={{ borderTop: '1px solid var(--line)' }}>
              <div className="mb-4 flex items-center gap-3">
                <div
                  className="flex h-7 w-7 items-center justify-center rounded-lg text-text-2"
                  style={{ border: '1px solid var(--line)', background: 'var(--bg-sunken)' }}
                >
                  <f.icon className="h-3.5 w-3.5" strokeWidth={1.6} />
                </div>
                <div className="mono text-[11px] tracking-[0.12em] text-text-3">{f.num}</div>
              </div>
              <h4 className="mb-1.5 text-[15.5px] font-semibold tracking-[-0.01em]">{f.title}</h4>
              <p className="max-w-[40ch] text-[13.5px] leading-relaxed text-text-2">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Resource entry */}
      <section className="py-20" style={{ borderTop: '1px solid var(--line)' }}>
        <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-3">
          {[
            { href: '/hot', icon: Library, title: locale === 'zh' ? '热题 50' : 'Hot 50', desc: locale === 'zh' ? '主流 AI 岗位高频面试题' : 'Top interview questions' },
            { href: '/bagu', icon: Library, title: locale === 'zh' ? '八股文' : 'Bagu', desc: locale === 'zh' ? '13 章理论问答专题' : '13 chapters of theory Q&A' },
            { href: '/interview', icon: Library, title: locale === 'zh' ? '模拟面试' : 'Mock Interview', desc: locale === 'zh' ? '随机抽题自测' : 'Random self-test' },
          ].map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="card card-hover flex items-center gap-4 p-5"
            >
              <span
                className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-accent"
                style={{ border: '1px solid var(--accent-line)', background: 'var(--accent-wash)' }}
              >
                <item.icon className="h-4.5 w-4.5" strokeWidth={1.6} />
              </span>
              <div className="flex-1">
                <h4 className="text-[14.5px] font-semibold tracking-[-0.01em]">{item.title}</h4>
                <p className="text-[12.5px] text-text-2">{item.desc}</p>
              </div>
              <ArrowRight className="h-4 w-4 text-text-3" />
            </Link>
          ))}
        </div>
      </section>
    </main>
  );
}
