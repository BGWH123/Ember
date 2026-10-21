import { useEffect, useMemo, useState } from 'react';
import { ArrowLeft, ArrowRight, BookOpen, Code2, ExternalLink, FileText, Languages } from 'lucide-react';

import { HandbookMarkdown } from '@/components/handbook/HandbookMarkdown';
import { WaitingState } from '@/components/ui/WaitingState';
import { useLocale } from '@/context/LocaleContext';
import {
  handbookAsset,
  handbookDocumentPath,
  handbookHref,
  loadHandbookIndex,
  type HandbookDocument,
  type HandbookIndex,
} from '@/lib/handbook';
import Link from '@/routing/Link';
import { useParams } from '@/routing/navigation';
import { appFetch } from '@/transport/client';

interface TocItem {
  depth: number;
  title: string;
  anchor: string;
}

function plainHeading(value: string) {
  return value.replace(/[*_`[\]]/g, '').replace(/\s+/g, ' ').trim();
}

function anchorOf(value: string) {
  return plainHeading(value)
    .toLowerCase()
    .replace(/[^\p{Letter}\p{Number}]+/gu, '-')
    .replace(/^-|-$/g, '');
}

function makeToc(markdown: string): TocItem[] {
  return [...markdown.matchAll(/^(#{2,3})\s+(.+)$/gm)].map((match) => ({
    depth: match[1].length,
    title: plainHeading(match[2]),
    anchor: anchorOf(match[2]),
  }));
}

function sourceFor(document: HandbookDocument, index: HandbookIndex) {
  return document.kind === 'question-bank'
    ? index.sources.find((source) => source.id === 'embodied-interview-qa')
    : index.sources.find((source) => source.id === 'aris-ai-offer');
}

export default function HandbookDetailPage() {
  const { kind, slug } = useParams<{ kind: string; slug: string }>();
  const { locale } = useLocale();
  const [index, setIndex] = useState<HandbookIndex | null>(null);
  const [markdown, setMarkdown] = useState('');
  const [codes, setCodes] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const document = index?.documents.find((item) => item.kind === kind && item.slug === slug);
  const source = document && index ? sourceFor(document, index) : undefined;

  useEffect(() => {
    setLoading(true);
    setError('');
    loadHandbookIndex()
      .then(setIndex)
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : String(reason));
        setLoading(false);
      });
  }, [kind, slug]);

  useEffect(() => {
    if (!index) return;
    const current = index.documents.find((item) => item.kind === kind && item.slug === slug);
    if (!current) {
      setError('未找到这篇理论手册内容');
      setLoading(false);
      return;
    }
    if (current.kind === 'blog') {
      setMarkdown('');
      setCodes({});
      setLoading(false);
      return;
    }
    const path = handbookDocumentPath(current, locale);
    if (!path) {
      setError('内容文件不存在');
      setLoading(false);
      return;
    }
    setLoading(true);
    appFetch(handbookAsset(path))
      .then((response) => {
        if (!response.ok) throw new Error(`正文加载失败 (${response.status})`);
        return response.text();
      })
      .then(setMarkdown)
      .then(async () => {
        const entries = await Promise.all((current.code ?? []).map(async (codePath) => {
          const response = await appFetch(handbookAsset(codePath));
          return [codePath, response.ok ? await response.text() : ''] as const;
        }));
        setCodes(Object.fromEntries(entries));
        setLoading(false);
      })
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : String(reason));
        setLoading(false);
      });
  }, [index, kind, locale, slug]);

  const toc = useMemo(() => makeToc(markdown), [markdown]);
  const siblings = index && document
    ? index.documents.filter((item) => item.category === document.category)
    : [];
  const siblingIndex = document ? siblings.findIndex((item) => item.slug === document.slug && item.kind === document.kind) : -1;
  const previous = siblingIndex > 0 ? siblings[siblingIndex - 1] : undefined;
  const next = siblingIndex >= 0 && siblingIndex < siblings.length - 1 ? siblings[siblingIndex + 1] : undefined;
  const zh = locale === 'zh';

  if (loading || (!index && !error)) return <WaitingState />;
  if (!index || !document) {
    return (
      <main className="desktop-page p-8">
        <Link href="/bagu" className="text-sm text-accent">← 返回理论手册</Link>
        <p role="alert" className="mt-5 text-hard">{error || '内容不存在'}</p>
      </main>
    );
  }

  const title = zh ? document.title : document.titleEn;
  const category = index.categories.find((item) => item.id === document.category);

  return (
    <div className="desktop-page bg-bg">
      <div className="max-w-[1420px] mx-auto grid grid-cols-[minmax(0,1fr)_230px] max-[1040px]:grid-cols-1">
        <main className="min-w-0 px-6 lg:px-10 py-7">
          <div className="mono text-xs text-text-3 flex items-center gap-2 mb-6">
            <Link href="/bagu" className="hover:text-accent transition-colors">{zh ? '理论手册' : 'Theory handbook'}</Link>
            <span>/</span>
            <span>{zh ? category?.title : category?.titleEn}</span>
          </div>

          <header className="pb-6 mb-7" style={{ borderBottom: '1px solid var(--line)' }}>
            <div className="flex items-center gap-2 text-[11px] uppercase tracking-[.08em] text-accent mb-3">
              {document.kind === 'tutorial' ? <BookOpen size={14} /> : document.kind === 'blog' ? <FileText size={14} /> : <Code2 size={14} />}
              <span>{document.kind === 'tutorial' ? (zh ? '双语教程' : 'Bilingual tutorial') : document.kind === 'blog' ? (zh ? '技术长文' : 'Technical blog') : (zh ? '具身题库' : 'Embodied question bank')}</span>
            </div>
            <h1 className="text-[clamp(28px,4vw,46px)] font-semibold tracking-[-0.035em] leading-tight text-text">{title}</h1>
            <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-text-3">
              {document.enPath ? <span className="inline-flex items-center gap-1.5"><Languages size={13} />中文 / English</span> : null}
              {document.questionCount ? <span>{document.questionCount} {zh ? '道面试题' : 'questions'}</span> : null}
              {document.code?.length ? <span>{document.code.length} {zh ? '个配套脚本' : 'code examples'}</span> : null}
            </div>
          </header>

          {error ? <p role="alert" className="mb-5 text-hard">{error}</p> : null}

          {document.kind === 'blog' && document.htmlPath ? (
            <div className="rounded-xl overflow-hidden border bg-white" style={{ borderColor: 'var(--line)', height: 'calc(100dvh - 230px)', minHeight: 640 }}>
              <iframe
                title={title}
                src={handbookAsset(document.htmlPath)}
                sandbox="allow-scripts allow-same-origin allow-popups"
                className="w-full h-full border-0"
              />
            </div>
          ) : (
            <HandbookMarkdown markdown={markdown} documents={index.documents} />
          )}

          {Object.entries(codes).length > 0 ? (
            <section className="mt-10 pt-7" style={{ borderTop: '1px solid var(--line)' }}>
              <div className="flex items-center gap-2 mb-4">
                <Code2 size={17} className="text-accent" />
                <h2 className="text-xl font-semibold text-text">{zh ? '配套可运行代码' : 'Runnable code'}</h2>
              </div>
              <div className="space-y-3">
                {Object.entries(codes).map(([path, code]) => (
                  <details key={path} className="rounded-xl border overflow-hidden" style={{ borderColor: 'var(--line)', background: 'var(--bg-elev)' }}>
                    <summary className="cursor-pointer px-4 py-3 mono text-xs text-text">{path.split('/').at(-1)}</summary>
                    <HandbookMarkdown markdown={`\`\`\`python\n${code}\n\`\`\``} documents={index.documents} />
                  </details>
                ))}
              </div>
            </section>
          ) : null}

          <nav className="mt-10 pt-6 grid grid-cols-2 gap-3" style={{ borderTop: '1px solid var(--line)' }}>
            {previous ? (
              <Link href={handbookHref(previous)} className="card card-hover p-4">
                <span className="text-[10px] text-text-3 inline-flex items-center gap-1"><ArrowLeft size={12} />{zh ? '上一篇' : 'Previous'}</span>
                <div className="mt-1.5 text-sm font-medium text-text">{zh ? previous.title : previous.titleEn}</div>
              </Link>
            ) : <span />}
            {next ? (
              <Link href={handbookHref(next)} className="card card-hover p-4 text-right">
                <span className="text-[10px] text-text-3 inline-flex items-center gap-1">{zh ? '下一篇' : 'Next'}<ArrowRight size={12} /></span>
                <div className="mt-1.5 text-sm font-medium text-text">{zh ? next.title : next.titleEn}</div>
              </Link>
            ) : null}
          </nav>
        </main>

        <aside className="sticky top-0 self-start h-[calc(100dvh-42px)] overflow-y-auto px-5 py-7 border-l max-[1040px]:hidden" style={{ borderColor: 'var(--line)', background: 'var(--bg-sunken)' }}>
          {toc.length > 0 ? (
            <>
              <div className="eyebrow mb-3">{zh ? '本页目录' : 'ON THIS PAGE'}</div>
              <nav className="space-y-0.5">
                {toc.map((item, indexValue) => (
                  <button
                    type="button"
                    key={`${item.anchor}-${indexValue}`}
                    onClick={() => globalThis.document.getElementById(item.anchor)?.scrollIntoView({ behavior: 'smooth', block: 'start' })}
                    className="block w-full text-left py-1.5 text-[11px] leading-snug text-text-3 hover:text-accent transition-colors"
                    style={{ paddingLeft: item.depth === 3 ? 12 : 0 }}
                  >
                    {item.title}
                  </button>
                ))}
              </nav>
            </>
          ) : null}

          {source ? (
            <div className="mt-7 pt-5" style={{ borderTop: '1px solid var(--line)' }}>
              <div className="eyebrow mb-3">{zh ? '来源与许可' : 'SOURCE & LICENSE'}</div>
              <p className="text-[11px] leading-relaxed text-text-3">
                {source.name}<br />
                {source.author}<br />
                MIT · {source.commit.slice(0, 8)}
              </p>
              <a href={source.url} target="_blank" rel="noreferrer" className="mt-3 inline-flex items-center gap-1 text-[11px] text-accent">
                {zh ? '查看上游仓库' : 'Upstream repository'}<ExternalLink size={10} />
              </a>
            </div>
          ) : null}
        </aside>
      </div>
    </div>
  );
}
