import { useEffect, useMemo, useState } from 'react';
import { BookOpen, Boxes, Code2, FileText, Languages, Search, ShieldCheck } from 'lucide-react';

import { WaitingState } from '@/components/ui/WaitingState';
import { useLocale } from '@/context/LocaleContext';
import {
  handbookHref,
  loadHandbookIndex,
  type HandbookDocument,
  type HandbookIndex,
  type HandbookKind,
} from '@/lib/handbook';
import Link from '@/routing/Link';

const kindLabels: Record<HandbookKind, { zh: string; en: string }> = {
  tutorial: { zh: '双语教程', en: 'Bilingual tutorial' },
  'question-bank': { zh: '题库', en: 'Question bank' },
  blog: { zh: '技术长文', en: 'Technical blog' },
};

function documentTitle(document: HandbookDocument, locale: 'zh' | 'en') {
  return locale === 'zh' ? document.title : document.titleEn;
}

export default function BaguPage() {
  const { locale } = useLocale();
  const [index, setIndex] = useState<HandbookIndex | null>(null);
  const [error, setError] = useState('');
  const [category, setCategory] = useState('');
  const [query, setQuery] = useState('');

  useEffect(() => {
    loadHandbookIndex().then(setIndex).catch((reason: unknown) => {
      setError(reason instanceof Error ? reason.message : String(reason));
    });
  }, []);

  const documents = useMemo(() => {
    if (!index) return [];
    const normalized = query.trim().toLowerCase();
    return index.documents.filter((document) => {
      if (category && document.category !== category) return false;
      if (!normalized) return true;
      const categoryInfo = index.categories.find((item) => item.id === document.category);
      return [
        document.title,
        document.titleEn,
        categoryInfo?.title,
        categoryInfo?.titleEn,
      ].some((value) => value?.toLowerCase().includes(normalized));
    });
  }, [category, index, query]);

  if (!index && !error) return <WaitingState />;
  if (!index) {
    return <main className="desktop-page p-8"><p role="alert" className="text-hard">{error}</p></main>;
  }

  const zh = locale === 'zh';
  return (
    <div className="desktop-page bg-bg">
      <main className="max-w-[1240px] mx-auto px-6 py-7">
        <section className="pb-7 mb-7" style={{ borderBottom: '1px solid var(--line)' }}>
          <div className="eyebrow mb-2">{zh ? '离线知识库' : 'OFFLINE KNOWLEDGE BASE'}</div>
          <div className="flex items-start justify-between gap-8 max-[760px]:flex-col">
            <div className="max-w-[760px]">
              <h1 className="text-[clamp(30px,4vw,48px)] font-semibold tracking-[-0.035em] leading-[1.08] text-text">
                {zh ? index.title : 'Ember AI Theory Handbook'}
              </h1>
              <p className="mt-3 text-[15px] leading-relaxed text-text-2">
                {zh ? index.subtitle : 'From foundations to LLM systems, generative models, agents and embodied AI.'}
              </p>
            </div>
            <Link
              href="/interview"
              className="shrink-0 flex items-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium text-white hover:opacity-90"
              style={{ background: 'var(--accent)' }}
            >
              <Boxes size={15} />
              {zh ? '随机抽题' : 'Practice questions'}
            </Link>
          </div>

          <div className="mt-7 grid grid-cols-2 md:grid-cols-5 gap-2.5">
            {[
              [BookOpen, index.stats.tutorials, zh ? '篇双语教程' : 'tutorials'],
              [FileText, index.stats.blogs, zh ? '篇技术长文' : 'technical blogs'],
              [Code2, index.stats.codeExamples, zh ? '个代码示例' : 'code examples'],
              [Boxes, index.stats.embodiedQuestions, zh ? '道具身题目' : 'embodied questions'],
              [Languages, index.stats.languages, zh ? '种语言' : 'languages'],
            ].map(([Icon, value, label]) => {
              const StatIcon = Icon as typeof BookOpen;
              return (
                <div key={String(label)} className="rounded-xl border px-3.5 py-3" style={{ borderColor: 'var(--line)', background: 'var(--bg-elev)' }}>
                  <StatIcon size={14} className="text-accent mb-2" />
                  <div className="text-lg font-semibold tabular-nums text-text">{String(value)}</div>
                  <div className="text-[11px] text-text-3">{String(label)}</div>
                </div>
              );
            })}
          </div>
        </section>

        <div className="relative mb-5">
          <Search size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-text-3" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={zh ? '搜索教程、方向或关键词…' : 'Search tutorials, categories or keywords…'}
            className="field w-full pl-10 pr-4 py-3"
          />
        </div>

        <div className="flex gap-2 overflow-x-auto pb-3 mb-5">
          <button
            type="button"
            onClick={() => setCategory('')}
            className="shrink-0 rounded-full px-3.5 py-1.5 text-xs font-medium transition-colors"
            style={category === '' ? { color: 'var(--accent)', background: 'var(--accent-wash)', border: '1px solid var(--accent-line)' } : { color: 'var(--text-2)', border: '1px solid var(--line)' }}
          >
            {zh ? '全部内容' : 'All'}
          </button>
          {index.categories.map((item) => (
            <button
              type="button"
              key={item.id}
              onClick={() => setCategory(item.id)}
              className="shrink-0 rounded-full px-3.5 py-1.5 text-xs font-medium transition-colors"
              style={category === item.id ? { color: 'var(--accent)', background: 'var(--accent-wash)', border: '1px solid var(--accent-line)' } : { color: 'var(--text-2)', border: '1px solid var(--line)' }}
            >
              {zh ? item.title : item.titleEn}
              <span className="ml-1.5 text-text-3">{index.documents.filter((document) => document.category === item.id).length}</span>
            </button>
          ))}
        </div>

        <div className="space-y-8">
          {index.categories.map((categoryInfo) => {
            const categoryDocuments = documents.filter((document) => document.category === categoryInfo.id);
            if (categoryDocuments.length === 0) return null;
            return (
              <section key={categoryInfo.id}>
                <div className="flex items-end justify-between gap-5 mb-3">
                  <div>
                    <h2 className="text-xl font-semibold tracking-[-0.02em] text-text">
                      {zh ? categoryInfo.title : categoryInfo.titleEn}
                    </h2>
                    <p className="mt-1 text-xs leading-relaxed text-text-3">{categoryInfo.description}</p>
                  </div>
                  <span className="mono text-[11px] text-text-3">{categoryDocuments.length} {zh ? '项' : 'items'}</span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
                  {categoryDocuments.map((document) => (
                    <Link
                      key={`${document.kind}:${document.slug}`}
                      href={handbookHref(document)}
                      className="card card-hover group min-h-[132px] p-4 flex flex-col"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span className="mono text-[10px] uppercase tracking-[.08em] text-accent">
                          {kindLabels[document.kind][locale]}
                        </span>
                        {document.questionCount ? (
                          <span className="mono text-[10px] text-text-3">{document.questionCount} Q</span>
                        ) : null}
                      </div>
                      <h3 className="mt-3 text-[15px] font-semibold leading-snug text-text group-hover:text-accent transition-colors">
                        {documentTitle(document, locale)}
                      </h3>
                      <div className="mt-auto pt-4 flex items-center gap-3 text-[11px] text-text-3">
                        {document.enPath ? <span>中 / EN</span> : null}
                        {document.code?.length ? <span>{document.code.length} code</span> : null}
                        <span className="ml-auto text-accent">{zh ? '打开 →' : 'Open →'}</span>
                      </div>
                    </Link>
                  ))}
                </div>
              </section>
            );
          })}
        </div>

        {documents.length === 0 ? (
          <div className="py-20 text-center text-sm text-text-3">{zh ? '没有匹配内容' : 'No matching content'}</div>
        ) : null}

        <footer className="mt-10 pt-5 flex items-start gap-3 text-xs leading-relaxed text-text-3" style={{ borderTop: '1px solid var(--line)' }}>
          <ShieldCheck size={16} className="mt-0.5 shrink-0 text-accent" />
          <p>
            {zh
              ? '内容已随 Ember 本地打包，可离线阅读。教程与博客来自 ARIS-in-AI-Offer，具身题库来自 embodied-interview-qa；原作者、固定版本与 MIT 许可均在每篇详情页和 NOTICE 中保留。'
              : 'Content is bundled locally for offline reading. Original authors, pinned revisions and MIT licenses are preserved on each detail page and in NOTICE.'}
          </p>
        </footer>
      </main>
    </div>
  );
}
