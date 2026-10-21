import { useState, type ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { Check, Copy } from 'lucide-react';
import 'katex/dist/katex.min.css';

import type { HandbookDocument } from '@/lib/handbook';

function textOf(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(textOf).join('');
  if (node && typeof node === 'object' && 'props' in node) {
    return textOf((node as { props: { children?: ReactNode } }).props.children);
  }
  return '';
}

function anchorOf(children: ReactNode) {
  return textOf(children)
    .toLowerCase()
    .replace(/[^\p{Letter}\p{Number}]+/gu, '-')
    .replace(/^-|-$/g, '');
}

function CodeBlock({ className, children }: { className?: string; children: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const code = textOf(children).replace(/\n$/, '');
  const language = className?.replace('language-', '') ?? 'text';

  async function copy() {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  }

  return (
    <div className="handbook-code">
      <div className="handbook-code__bar">
        <span>{language}</span>
        <button type="button" onClick={() => void copy()} aria-label="复制代码">
          {copied ? <Check size={13} /> : <Copy size={13} />}
          {copied ? '已复制' : '复制'}
        </button>
      </div>
      <pre><code className={className}>{code}</code></pre>
    </div>
  );
}

export function HandbookMarkdown({
  markdown,
  documents,
}: {
  markdown: string;
  documents: HandbookDocument[];
}) {
  function hrefFor(href?: string) {
    if (!href) return '';
    if (/^(https?:|mailto:)/i.test(href) || href.startsWith('#')) return href;
    const file = href.split('/').at(-1)?.split('#')[0];
    if (file?.endsWith('.py')) return `/handbook/code/${file}`;
    const target = documents.find((document) =>
      document.sourceFile === file || document.sourceFileEn === file
    );
    return target ? `#${`/bagu/${target.kind}/${target.slug}`}` : href;
  }

  return (
    <article className="handbook-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeRaw, rehypeKatex]}
        components={{
          h1: ({ children }) => <h1 id={anchorOf(children)}>{children}</h1>,
          h2: ({ children }) => <h2 id={anchorOf(children)}>{children}</h2>,
          h3: ({ children }) => <h3 id={anchorOf(children)}>{children}</h3>,
          h4: ({ children }) => <h4 id={anchorOf(children)}>{children}</h4>,
          pre: ({ children }) => <>{children}</>,
          code: ({ className, children }) =>
            className ? (
              <CodeBlock className={className}>{children}</CodeBlock>
            ) : (
              <code>{children}</code>
            ),
          a: ({ href, children }) => {
            const resolved = hrefFor(href);
            const external = /^(https?:|mailto:)/i.test(resolved);
            if (resolved.startsWith('#') && !resolved.startsWith('#/')) {
              return (
                <a
                  href={resolved}
                  onClick={(event) => {
                    event.preventDefault();
                    globalThis.document.getElementById(resolved.slice(1))?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                  }}
                >
                  {children}
                </a>
              );
            }
            return (
              <a href={resolved} target={external ? '_blank' : undefined} rel={external ? 'noreferrer' : undefined}>
                {children}
              </a>
            );
          },
        }}
      >
        {markdown}
      </ReactMarkdown>
    </article>
  );
}
