'use client';

import { useState, useEffect, useMemo, useCallback } from 'react';
import { TopNav } from '@/components/layout/TopNav';
import { Footer } from '@/components/layout/Footer';
import {
  Search,
  BookOpen,
  ChevronRight,
  ChevronDown,
  MessageCircle,
  ArrowUp,
  Eye,
  EyeOff,
} from 'lucide-react';
import Link from 'next/link';

// ─── Types ───────────────────────────────────────────────────────────────

interface SubSubSection {
  title: string;
  content: string;
}

interface SubSection {
  title: string;
  content: string;
  subsubsections: SubSubSection[];
}

interface Section {
  title: string;
  content: string;
  subsections: SubSection[];
}

interface Chapter {
  id: string;
  title: string;
  sections: Section[];
}

// ─── Data Hook ───────────────────────────────────────────────────────────

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

// ─── Helpers ─────────────────────────────────────────────────────────────

function cn(...classes: (string | false | undefined)[]) {
  return classes.filter(Boolean).join(' ');
}

function countQuestions(chapters: Chapter[]) {
  return chapters.reduce((acc, ch) => acc + ch.sections.length, 0);
}

// ─── Interview Link Banner ───────────────────────────────────────────────

function InterviewBanner() {
  return (
    <Link
      href="/interview"
      className="group flex items-center gap-3 px-4 py-3 rounded-xl border text-left transition-all duration-200 hover:shadow-sm hover:border-accent mb-6"
      style={{ borderColor: 'var(--line)', background: 'var(--bg-card)' }}
    >
      <div
        className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0 transition-colors group-hover:bg-accent-wash"
        style={{ background: 'var(--accent-bg)' }}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-accent">
          <path d="M7 17L17 7" />
          <path d="M7 7h10v10" />
        </svg>
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-sm font-semibold text-text">进入模拟面试</div>
        <div className="text-xs text-text-3 mt-0.5">随机抽题自测 · 八股面试 + 代码面试</div>
      </div>
      <ArrowUp size={14} className="text-text-3 group-hover:text-accent transition-colors rotate-45" />
    </Link>
  );
}

// ─── Sidebar ─────────────────────────────────────────────────────────────

function Sidebar({
  chapters,
  activeChapter,
  activeSection,
  onSelect,
  expandedChapters,
  toggleChapter,
}: {
  chapters: Chapter[];
  activeChapter: string;
  activeSection: string;
  onSelect: (chId: string, secKey: string) => void;
  expandedChapters: Set<string>;
  toggleChapter: (id: string) => void;
}) {
  return (
    <aside
      className="w-[280px] shrink-0 h-[calc(100vh-60px)] overflow-y-auto border-r hidden md:block"
      style={{ borderColor: 'var(--line)', background: 'var(--bg-sunken)' }}
    >
      <div className="p-4">
        {/* Interview Entry */}
        <Link
          href="/interview"
          className="flex items-center gap-2 w-full px-2.5 py-2 rounded-lg mb-4 text-sm font-medium transition-all duration-200 hover:opacity-90"
          style={{
            background: 'var(--accent-bg)',
            color: 'var(--accent)',
            border: '1px solid var(--accent-line)',
          }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M7 17L17 7" />
            <path d="M7 7h10v10" />
          </svg>
          <span>模拟面试</span>
        </Link>

        <div className="flex items-center gap-2 mb-3 px-2">
          <BookOpen size={16} className="text-accent" />
          <span className="text-sm font-semibold text-text">目录</span>
        </div>
        {chapters.map((ch) => {
          const isExpanded = expandedChapters.has(ch.id);
          const isActive = activeChapter === ch.id;
          return (
            <div key={ch.id} className="mb-1">
              <button
                onClick={() => toggleChapter(ch.id)}
                className={cn(
                  'w-full flex items-center gap-1.5 px-2 py-1.5 rounded-lg text-left text-sm transition-all duration-200',
                  isActive
                    ? 'text-accent font-medium'
                    : 'text-text-2 hover:text-text'
                )}
                style={isActive ? { background: 'var(--accent-bg)' } : {}}
              >
                {isExpanded ? (
                  <ChevronDown size={13} />
                ) : (
                  <ChevronRight size={13} />
                )}
                <span className="truncate">{ch.title}</span>
              </button>
              {isExpanded && (
                <div className="ml-4 mt-0.5 space-y-0.5">
                  {ch.sections.map((sec, idx) => {
                    const secKey = `${ch.id}::${idx}`;
                    const isSecActive = activeSection === secKey;
                    return (
                      <button
                        key={secKey}
                        onClick={() => onSelect(ch.id, secKey)}
                        className={cn(
                          'w-full text-left px-2 py-1 rounded-md text-xs transition-colors truncate',
                          isSecActive
                            ? 'text-accent font-medium'
                            : 'text-text-3 hover:text-text-2'
                        )}
                        style={
                          isSecActive ? { background: 'var(--accent-bg)' } : {}
                        }
                        title={sec.title}
                      >
                        {sec.title || '(无标题)'}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </aside>
  );
}

// ─── RichText: render **bold** inline ────────────────────────────────────

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

// ─── RevealBlock: per-item collapsible ───────────────────────────────────

function RevealBlock({
  title,
  children,
}: {
  title?: string;
  children: React.ReactNode;
}) {
  const [revealed, setRevealed] = useState(false);

  if (!revealed) {
    return (
      <div
        className="rounded-lg border cursor-pointer transition-all duration-200 hover:border-accent hover:shadow-sm"
        style={{ borderColor: 'var(--line)', background: 'var(--bg-sunken)' }}
        onClick={() => setRevealed(true)}
      >
        <div className="flex items-center justify-between px-3.5 py-2.5">
          <span className="text-sm font-medium text-text truncate pr-3">
            {title ? <RichText text={title} /> : <span className="text-text-3 font-normal">答案</span>}
          </span>
          <div
            className="flex items-center gap-1 text-[11px] font-medium shrink-0 px-2 py-0.5 rounded-full"
            style={{ color: 'var(--accent)', background: 'var(--accent-bg)' }}
          >
            <Eye size={11} />
            <span>显示</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className="rounded-lg border p-3.5"
      style={{ borderColor: 'var(--line)', background: 'var(--bg-sunken)' }}
    >
      {title && (
        <h4 className="text-sm font-medium text-text mb-1.5">
          <RichText text={title} />
        </h4>
      )}
      {children}
      <button
        onClick={() => setRevealed(false)}
        className="mt-2 flex items-center gap-1 text-[11px] text-text-3 hover:text-accent transition-colors"
      >
        <EyeOff size={11} />
        <span>收起</span>
      </button>
    </div>
  );
}

// ─── ContentCard: each subsection independently collapsible ──────────────

function ContentCard({
  title,
  content,
  subsections,
  id,
}: {
  title: string;
  content: string;
  subsections: SubSection[];
  id: string;
}) {
  if (!title && !content && subsections.length === 0) return null;

  return (
    <div
      id={id}
      className="mb-5 rounded-xl border p-5 transition-all duration-300 hover:shadow-md"
      style={{
        borderColor: 'var(--line)',
        background: 'var(--bg-card)',
      }}
    >
      {/* Question Title */}
      {title && (
        <h3 className="text-base font-semibold text-text mb-3 flex items-start gap-2.5">
          <MessageCircle size={16} className="text-accent mt-1 shrink-0" />
          <RichText text={title} />
        </h3>
      )}

      {/* Content block (if no subsections) */}
      {content && subsections.length === 0 && (
        <RevealBlock>
          <div className="text-sm text-text-2 leading-relaxed whitespace-pre-wrap">
            <RichText text={content} />
          </div>
        </RevealBlock>
      )}

      {/* Content shown directly + subsections collapsed */}
      {content && subsections.length > 0 && (
        <div className="text-sm text-text-2 leading-relaxed whitespace-pre-wrap mb-3">
          <RichText text={content} />
        </div>
      )}

      {/* Subsection blocks */}
      {subsections.length > 0 && (
        <div className="space-y-2.5">
          {subsections.map((sub, i) => (
            <RevealBlock key={i} title={sub.title}>
              <div className="text-xs text-text-2 leading-relaxed whitespace-pre-wrap">
                <RichText text={sub.content} />
              </div>
              {sub.subsubsections?.length > 0 && (
                <div className="mt-2 space-y-1.5">
                  {sub.subsubsections.map((ss, j) => (
                    <div
                      key={j}
                      className="ml-2 pl-3 border-l-2"
                      style={{ borderColor: 'var(--accent)' }}
                    >
                      {ss.title && (
                        <h5 className="text-xs font-medium text-text">
                          <RichText text={ss.title} />
                        </h5>
                      )}
                      {ss.content && (
                        <div className="text-xs text-text-2 leading-relaxed whitespace-pre-wrap mt-0.5">
                          <RichText text={ss.content} />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </RevealBlock>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────────────────────

export default function BaguPage() {
  const { chapters, loading } = useBaguData();
  const [searchQuery, setSearchQuery] = useState('');
  const [activeChapter, setActiveChapter] = useState('');
  const [activeSection, setActiveSection] = useState('');
  const [expandedChapters, setExpandedChapters] = useState<Set<string>>(
    new Set()
  );

  // Initialize expanded chapters
  useEffect(() => {
    if (chapters.length > 0) {
      const ids = new Set(chapters.map((c) => c.id));
      setExpandedChapters(ids);
      setActiveChapter(chapters[0].id);
    }
  }, [chapters.length]);

  const filteredChapters = useMemo(() => {
    if (!searchQuery.trim()) return chapters;
    const q = searchQuery.toLowerCase();
    return chapters
      .map((ch) => ({
        ...ch,
        sections: ch.sections.filter(
          (sec) =>
            sec.title.toLowerCase().includes(q) ||
            sec.content.toLowerCase().includes(q) ||
            sec.subsections.some(
              (sub) =>
                sub.title.toLowerCase().includes(q) ||
                sub.content.toLowerCase().includes(q)
            )
        ),
      }))
      .filter((ch) => ch.sections.length > 0);
  }, [chapters, searchQuery]);

  const handleSelect = useCallback((chId: string, secKey: string) => {
    setActiveChapter(chId);
    setActiveSection(secKey);
    const el = document.getElementById(secKey);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, []);

  const toggleChapter = useCallback((id: string) => {
    setExpandedChapters((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const scrollToTop = () => window.scrollTo({ top: 0, behavior: 'smooth' });

  if (loading) {
    return (
      <div className="min-h-screen bg-bg flex items-center justify-center">
        <div className="text-text-2 text-sm">加载中...</div>
      </div>
    );
  }

  const activeCh = chapters.find((c) => c.id === activeChapter);

  return (
    <div className="min-h-screen bg-bg">
      <TopNav />
      <div className="flex" style={{ marginTop: 60 }}>
        <Sidebar
          chapters={filteredChapters}
          activeChapter={activeChapter}
          activeSection={activeSection}
          onSelect={handleSelect}
          expandedChapters={expandedChapters}
          toggleChapter={toggleChapter}
        />
        <main className="flex-1 min-h-[calc(100vh-60px)]">
          <div className="max-w-[800px] mx-auto px-6 py-6">
            {/* Header */}
            <div className="mb-6">
              <h1 className="text-xl font-bold text-text mb-2">
                大模型面试八股文
              </h1>
              <p className="text-sm text-text-2">
                覆盖基础面、进阶面、评测面、推理面、微调面、RLHF、RAG、Agent
                等 13 个方向，共 {countQuestions(chapters)}+ 个高频问题
              </p>
            </div>

            {/* Interview Link */}
            <InterviewBanner />

            {/* Search */}
            <div className="relative mb-6">
              <Search
                size={15}
                className="absolute left-3 top-1/2 -translate-y-1/2 text-text-3"
              />
              <input
                type="text"
                placeholder="搜索问题或关键词..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-9 pr-4 py-2.5 rounded-xl text-sm border outline-none focus:border-accent transition-colors"
                style={{
                  background: 'var(--bg-sunken)',
                  borderColor: 'var(--line)',
                  color: 'var(--text)',
                }}
              />
            </div>

            {/* Content */}
            {searchQuery.trim() ? (
              // Search results across all chapters
              <div>
                <div className="text-xs text-text-3 mb-3">
                  找到{' '}
                  {filteredChapters.reduce(
                    (acc, ch) => acc + ch.sections.length,
                    0
                  )}{' '}
                  条结果
                </div>
                {filteredChapters.map((ch) =>
                  ch.sections.map((sec, idx) => (
                    <ContentCard
                      key={`${ch.id}::${idx}`}
                      id={`${ch.id}::${idx}`}
                      title={`[${ch.title}] ${sec.title}`}
                      content={sec.content}
                      subsections={sec.subsections}
                    />
                  ))
                )}
              </div>
            ) : activeCh ? (
              // Show active chapter content
              <div>
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-lg font-semibold text-text">
                    {activeCh.title}
                  </span>
                  <span
                    className="text-xs text-text-3 px-2 py-0.5 rounded-full"
                    style={{ background: 'var(--bg-sunken)' }}
                  >
                    {activeCh.sections.length} 个问题
                  </span>
                </div>
                {activeCh.sections.map((sec, idx) => (
                  <ContentCard
                    key={idx}
                    id={`${activeCh.id}::${idx}`}
                    title={sec.title}
                    content={sec.content}
                    subsections={sec.subsections}
                  />
                ))}
              </div>
            ) : (
              <div className="text-center py-20 text-text-3 text-sm">
                选择一个章节开始阅读
              </div>
            )}
          </div>
        </main>
      </div>
      <Footer />

      {/* Back to top */}
      <button
        onClick={scrollToTop}
        className="fixed bottom-6 right-6 p-2.5 rounded-full border shadow-sm hover:shadow-md transition-shadow"
        style={{ background: 'var(--bg-card)', borderColor: 'var(--line)' }}
        title="回到顶部"
      >
        <ArrowUp size={16} className="text-text-2" />
      </button>
    </div>
  );
}
