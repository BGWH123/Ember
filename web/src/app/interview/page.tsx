'use client';

import { useState, useEffect, useMemo, useCallback } from 'react';
import { TopNav } from '@/components/layout/TopNav';
import { Footer } from '@/components/layout/Footer';
import {
  Dices,
  Code2,
  Shuffle,
  Eye,
  EyeOff,
  ArrowRight,
  MessageSquare,
  Terminal,
  Sparkles,
  RotateCcw,
} from 'lucide-react';
import { useRouter } from 'next/navigation';

// ─── Types ───────────────────────────────────────────────────────────────

interface Section {
  title: string;
  content: string;
  subsections: { title: string; content: string; subsubsections?: { title: string; content: string }[] }[];
}

interface Chapter {
  id: string;
  title: string;
  sections: Section[];
}

interface ProblemItem {
  id: string;
  title: string;
  titleZh: string;
  difficulty: string;
  category: string;
}

// ─── Data Hooks ──────────────────────────────────────────────────────────

function useBaguData() {
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/bagu-data.json')
      .then((r) => r.json())
      .then((d) => {
        setChapters(d || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  return { chapters, loading };
}

function useProblems() {
  const [problems, setProblems] = useState<ProblemItem[]>([]);

  useEffect(() => {
    fetch('/problems-list.json')
      .then((r) => r.json())
      .then((d) => setProblems(d.problems || []))
      .catch(() => setProblems([]));
  }, []);

  return problems;
}

// ─── Helpers ─────────────────────────────────────────────────────────────

function cn(...classes: (string | false | undefined)[]) {
  return classes.filter(Boolean).join(' ');
}

function RichText({ text, className }: { text: string; className?: string }) {
  if (!text) return null;
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return (
    <span className={className}>
      {parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return (
            <strong key={i} className="font-semibold text-text">
              {part.slice(2, -2)}
            </strong>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </span>
  );
}

function difficultyColor(d: string) {
  switch (d) {
    case 'Easy':
      return 'text-green-600 bg-green-50 dark:bg-green-900/20';
    case 'Medium':
      return 'text-amber-600 bg-amber-50 dark:bg-amber-900/20';
    case 'Hard':
      return 'text-red-600 bg-red-50 dark:bg-red-900/20';
    default:
      return 'text-text-3 bg-bg-sunken';
  }
}

// ─── Bagu Quiz Module ────────────────────────────────────────────────────

function BaguQuizModule({ chapters }: { chapters: Chapter[] }) {
  const allSections = useMemo(() => {
    const list: { chapterTitle: string; section: Section }[] = [];
    for (const ch of chapters) {
      for (const sec of ch.sections) {
        list.push({ chapterTitle: ch.title, section: sec });
      }
    }
    return list;
  }, [chapters]);

  const [current, setCurrent] = useState<{ chapterTitle: string; section: Section } | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [started, setStarted] = useState(false);
  const [history, setHistory] = useState<number[]>([]);

  const draw = useCallback(() => {
    if (allSections.length === 0) return;
    let idx: number;
    do {
      idx = Math.floor(Math.random() * allSections.length);
    } while (history.includes(idx) && history.length < allSections.length);
    setCurrent(allSections[idx]);
    setRevealed(false);
    setStarted(true);
    setHistory((prev) => [...prev.slice(-19), idx]);
  }, [allSections, history]);

  const reset = () => {
    setCurrent(null);
    setRevealed(false);
    setStarted(false);
    setHistory([]);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Module Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2.5">
          <div
            className="w-9 h-9 rounded-lg flex items-center justify-center"
            style={{ background: 'var(--accent-bg)' }}
          >
            <MessageSquare size={18} className="text-accent" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-text">八股面试</h2>
            <p className="text-xs text-text-3">
              {allSections.length > 0 ? `共 ${allSections.length} 道高频面试题` : '加载中...'}
            </p>
          </div>
        </div>
        {started && (
          <button
            onClick={reset}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs text-text-3 hover:text-text transition-colors"
            style={{ border: '1px solid var(--line)' }}
          >
            <RotateCcw size={12} />
            重置
          </button>
        )}
      </div>

      {/* Quiz Area */}
      {!started ? (
        <div
          className="flex-1 flex flex-col items-center justify-center rounded-2xl border p-8 text-center"
          style={{ borderColor: 'var(--line)', background: 'var(--bg-card)' }}
        >
          <div
            className="w-14 h-14 rounded-2xl flex items-center justify-center mb-4"
            style={{ background: 'var(--accent-bg)' }}
          >
            <Dices size={28} className="text-accent" />
          </div>
          <h3 className="text-sm font-semibold text-text mb-1">准备开始八股面试</h3>
          <p className="text-xs text-text-3 mb-5 max-w-[240px]">
            随机抽取面试题，先自己思考回答，再点击查看参考答案
          </p>
          <button
            onClick={draw}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium text-white transition-all hover:opacity-90 active:scale-95"
            style={{ background: 'var(--accent)' }}
          >
            <Sparkles size={14} />
            开始抽题
          </button>
        </div>
      ) : (
        <div className="flex-1 flex flex-col">
          {/* Question Card */}
          <div
            className="flex-1 rounded-2xl border p-5 flex flex-col"
            style={{ borderColor: 'var(--line)', background: 'var(--bg-card)' }}
          >
            {current ? (
              <>
                <div className="flex items-center gap-2 mb-3">
                  <span
                    className="text-[11px] text-text-3 px-2 py-0.5 rounded-md"
                    style={{ background: 'var(--bg-sunken)' }}
                  >
                    {current.chapterTitle}
                  </span>
                  <span className="text-[11px] text-text-3">
                    第 {history.length} 题
                  </span>
                </div>

                <h3 className="text-[15px] font-semibold text-text mb-4 leading-relaxed">
                  <RichText text={current.section.title} />
                </h3>

                {/* Answer Area */}
                <div className="flex-1 relative rounded-xl overflow-hidden" style={{ border: '1px solid var(--line)' }}>
                  {!revealed && (
                    <div
                      className="absolute inset-0 z-10 flex flex-col items-center justify-center cursor-pointer"
                      style={{
                        background: 'var(--bg-card)',
                      }}
                      onClick={() => setRevealed(true)}
                    >
                      <div
                        className="flex items-center gap-2 px-5 py-2.5 rounded-full text-sm font-medium transition-all duration-200 hover:scale-105"
                        style={{
                          background: 'var(--accent-bg)',
                          color: 'var(--accent)',
                          border: '1px solid var(--accent-line)',
                        }}
                      >
                        <Eye size={14} />
                        <span>显示答案</span>
                      </div>
                      <p className="text-xs text-text-3 mt-3">先自己组织语言回答，再对照参考答案</p>
                    </div>
                  )}

                  <div
                    className={cn(
                      'p-4 text-sm text-text-2 leading-relaxed whitespace-pre-wrap transition-all duration-500 overflow-y-auto max-h-[360px]',
                      revealed ? 'opacity-100' : 'opacity-10 blur-[1px] select-none'
                    )}
                  >
                    <RichText text={current.section.content} />
                    {current.section.subsections.map((sub, i) => (
                      <div key={i} className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--line)' }}>
                        {sub.title && (
                          <div className="font-medium text-text text-xs mb-1">
                            <RichText text={sub.title} />
                          </div>
                        )}
                        <div className="text-xs">
                          <RichText text={sub.content} />
                        </div>
                        {sub.subsubsections?.map((ss, j) => (
                          <div key={j} className="mt-2 ml-2 pl-3 border-l-2" style={{ borderColor: 'var(--accent)' }}>
                            {ss.title && <div className="font-medium text-text text-[11px]"><RichText text={ss.title} /></div>}
                            <div className="text-[11px] text-text-2"><RichText text={ss.content} /></div>
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center text-text-3 text-sm">
                数据加载中...
              </div>
            )}
          </div>

          {/* Action Bar */}
          <div className="flex items-center justify-between mt-4">
            <div className="flex items-center gap-2">
              {revealed && (
                <button
                  onClick={() => setRevealed(false)}
                  className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs text-text-3 hover:text-text transition-colors"
                  style={{ border: '1px solid var(--line)' }}
                >
                  <EyeOff size={12} />
                  隐藏答案
                </button>
              )}
            </div>
            <button
              onClick={draw}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-medium text-white transition-all hover:opacity-90 active:scale-95"
              style={{ background: 'var(--accent)' }}
            >
              <Shuffle size={12} />
              下一题
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Code Quiz Module ────────────────────────────────────────────────────

function CodeQuizModule({ problems }: { problems: ProblemItem[] }) {
  const router = useRouter();
  const [current, setCurrent] = useState<ProblemItem | null>(null);
  const [started, setStarted] = useState(false);
  const [history, setHistory] = useState<number[]>([]);

  const draw = useCallback(() => {
    if (problems.length === 0) return;
    let idx: number;
    do {
      idx = Math.floor(Math.random() * problems.length);
    } while (history.includes(idx) && history.length < problems.length);
    setCurrent(problems[idx]);
    setStarted(true);
    setHistory((prev) => [...prev.slice(-19), idx]);
  }, [problems, history]);

  const reset = () => {
    setCurrent(null);
    setStarted(false);
    setHistory([]);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Module Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2.5">
          <div
            className="w-9 h-9 rounded-lg flex items-center justify-center"
            style={{ background: 'var(--accent-bg)' }}
          >
            <Terminal size={18} className="text-accent" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-text">代码面试</h2>
            <p className="text-xs text-text-3">
              {problems.length > 0 ? `共 ${problems.length} 道手撕代码题` : '加载中...'}
            </p>
          </div>
        </div>
        {started && (
          <button
            onClick={reset}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs text-text-3 hover:text-text transition-colors"
            style={{ border: '1px solid var(--line)' }}
          >
            <RotateCcw size={12} />
            重置
          </button>
        )}
      </div>

      {/* Quiz Area */}
      {!started ? (
        <div
          className="flex-1 flex flex-col items-center justify-center rounded-2xl border p-8 text-center"
          style={{ borderColor: 'var(--line)', background: 'var(--bg-card)' }}
        >
          <div
            className="w-14 h-14 rounded-2xl flex items-center justify-center mb-4"
            style={{ background: 'var(--accent-bg)' }}
          >
            <Code2 size={28} className="text-accent" />
          </div>
          <h3 className="text-sm font-semibold text-text mb-1">准备开始代码面试</h3>
          <p className="text-xs text-text-3 mb-5 max-w-[240px]">
            随机抽取手撕代码题，直接跳转到在线刷题环境
          </p>
          <button
            onClick={draw}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium text-white transition-all hover:opacity-90 active:scale-95"
            style={{ background: 'var(--accent)' }}
          >
            <Sparkles size={14} />
            开始抽题
          </button>
        </div>
      ) : (
        <div className="flex-1 flex flex-col">
          {/* Question Card */}
          <div
            className="flex-1 rounded-2xl border p-5 flex flex-col"
            style={{ borderColor: 'var(--line)', background: 'var(--bg-card)' }}
          >
            {current ? (
              <>
                <div className="flex items-center gap-2 mb-4">
                  <span
                    className={cn(
                      'text-[10px] font-semibold px-2 py-0.5 rounded-full uppercase tracking-wider',
                      difficultyColor(current.difficulty)
                    )}
                  >
                    {current.difficulty}
                  </span>
                  <span
                    className="text-[10px] text-text-3 px-2 py-0.5 rounded-full"
                    style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}
                  >
                    {current.category}
                  </span>
                  <span className="text-[11px] text-text-3 ml-auto">
                    第 {history.length} 题
                  </span>
                </div>

                <h3 className="text-[15px] font-semibold text-text mb-1">
                  {current.titleZh || current.title}
                </h3>
                <p className="text-sm text-text-3 mb-6">{current.title}</p>

                <div className="mt-auto flex items-center gap-3">
                  <button
                    onClick={() => router.push(`/problems/${current.id}`)}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium text-white transition-all hover:opacity-90 active:scale-95"
                    style={{ background: 'var(--accent)' }}
                  >
                    <Code2 size={14} />
                    去刷题
                    <ArrowRight size={14} />
                  </button>
                  <button
                    onClick={draw}
                    className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs text-text-3 hover:text-text transition-colors"
                    style={{ border: '1px solid var(--line)' }}
                  >
                    <Shuffle size={12} />
                    换一题
                  </button>
                </div>
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center text-text-3 text-sm">
                数据加载中...
              </div>
            )}
          </div>

          {/* Stats */}
          <div className="mt-4 grid grid-cols-3 gap-2">
            {['Easy', 'Medium', 'Hard'].map((d) => {
              const count = problems.filter((p) => p.difficulty === d).length;
              return (
                <div
                  key={d}
                  className="rounded-lg border px-3 py-2 text-center"
                  style={{ borderColor: 'var(--line)', background: 'var(--bg-sunken)' }}
                >
                  <div className="text-xs font-semibold text-text">{count}</div>
                  <div className="text-[10px] text-text-3">{d}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────────────────────

export default function InterviewPage() {
  const { chapters, loading } = useBaguData();
  const problems = useProblems();

  if (loading) {
    return (
      <div className="min-h-screen bg-bg flex items-center justify-center">
        <div className="text-text-2 text-sm">加载中...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-bg">
      <TopNav />

      <main className="max-w-[1200px] mx-auto px-6" style={{ paddingTop: 80 }}>
        {/* Page Header */}
        <div className="text-center mb-10">
          <h1 className="text-2xl font-bold text-text mb-2">模拟面试</h1>
          <p className="text-sm text-text-2 max-w-[480px] mx-auto">
            随机抽题模拟真实面试场景，八股面试先思考再看答案，代码面试直接上手实战
          </p>
        </div>

        {/* Two Modules */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pb-12">
          {/* Bagu Module */}
          <div
            className="rounded-2xl border p-6 flex flex-col"
            style={{ borderColor: 'var(--line)', background: 'var(--bg-elev)', minHeight: 560 }}
          >
            <BaguQuizModule chapters={chapters} />
          </div>

          {/* Code Module */}
          <div
            className="rounded-2xl border p-6 flex flex-col"
            style={{ borderColor: 'var(--line)', background: 'var(--bg-elev)', minHeight: 560 }}
          >
            <CodeQuizModule problems={problems} />
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
