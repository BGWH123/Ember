'use client';

import { useState } from 'react';
import { Lightbulb, ChevronDown, ChevronRight, BookOpen, GitBranch } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { useLocale } from '@/context/LocaleContext';
import type { Problem } from '@/lib/types';
import { RichText } from './MathRenderer';

function parseInline(text: string): (string | JSX.Element)[] {
  const parts: (string | JSX.Element)[] = [];
  let remaining = text;
  let key = 0;

  while (remaining.length > 0) {
    const boldMatch = remaining.match(/\*\*(.+?)\*\*/);
    const codeMatch = remaining.match(/`(.+?)`/);
    const latexMatch = remaining.match(/\$\$(.+?)\$\$/);
    const inlineLatexMatch = remaining.match(/\$(.+?)\$/);

    const boldIdx = boldMatch?.index ?? Infinity;
    const codeIdx = codeMatch?.index ?? Infinity;
    const latexIdx = latexMatch?.index ?? Infinity;
    const inlineLatexIdx = inlineLatexMatch?.index ?? Infinity;

    if (boldIdx === Infinity && codeIdx === Infinity && latexIdx === Infinity && inlineLatexIdx === Infinity) {
      parts.push(remaining);
      break;
    }

    const firstIdx = Math.min(boldIdx, codeIdx, latexIdx, inlineLatexIdx);

    if (firstIdx > 0) parts.push(remaining.slice(0, firstIdx));

    if (boldIdx === firstIdx && boldMatch) {
      parts.push(<strong key={key++} className="font-medium text-text">{boldMatch[1]}</strong>);
      remaining = remaining.slice(boldIdx + boldMatch[0].length);
    } else if (codeIdx === firstIdx && codeMatch) {
      parts.push(
        <code
          key={key++}
          className="mono text-[12.5px] px-[5px] py-px rounded text-text"
          style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}
        >
          {codeMatch[1]}
        </code>
      );
      remaining = remaining.slice(codeIdx + codeMatch[0].length);
    } else if (latexIdx === firstIdx && latexMatch) {
      parts.push(
        <div
          key={key++}
          className="my-2 px-3 py-2 rounded text-sm overflow-x-auto"
          style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}
        >
          <code className="mono text-[13px] text-text">{latexMatch[1]}</code>
        </div>
      );
      remaining = remaining.slice(latexIdx + latexMatch[0].length);
    } else if (inlineLatexIdx === firstIdx && inlineLatexMatch) {
      parts.push(
        <code
          key={key++}
          className="mono text-[13px] px-1 py-0.5 rounded text-text"
          style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}
        >
          {inlineLatexMatch[1]}
        </code>
      );
      remaining = remaining.slice(inlineLatexIdx + inlineLatexMatch[0].length);
    } else {
      parts.push(remaining);
      break;
    }
  }
  return parts;
}

function renderDescription(text: string) {
  return text.split('\n').map((line, i) => {
    if (line === '') return <div key={i} className="h-2" />;
    if (line.startsWith('- ')) {
      return <li key={i} className="ml-4 list-disc text-sm text-text-2 leading-relaxed">{parseInline(line.slice(2))}</li>;
    }
    if (line.startsWith('| ')) {
      return <div key={i} className="text-sm text-text-2 font-mono leading-relaxed">{line}</div>;
    }
    return <p key={i} className="text-sm text-text-2 leading-relaxed">{parseInline(line)}</p>;
  });
}

function renderMermaid(text: string) {
  const lines = text.split('\n').filter((l) => l.trim());
  return (
    <div
      className="my-2 p-3 rounded text-sm overflow-x-auto"
      style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}
    >
      <pre className="mono text-[12.5px] text-text-2 leading-relaxed">
        {lines.map((line, i) => (
          <div key={i}>{line}</div>
        ))}
      </pre>
    </div>
  );
}

interface DescriptionTabProps {
  problem: Problem;
}

export function DescriptionTab({ problem }: DescriptionTabProps) {
  const [hintOpen, setHintOpen] = useState(false);
  const [theoryOpen, setTheoryOpen] = useState(true);
  const [diagramOpen, setDiagramOpen] = useState(true);
  const { locale, t } = useLocale();

  const description = locale === 'zh' ? problem.descriptionZh : problem.descriptionEn;
  const hint = locale === 'zh' && problem.hintZh ? problem.hintZh : problem.hint;
  const theory = locale === 'zh' ? problem.theoryZh : problem.theoryEn;
  const diagram = locale === 'zh' ? problem.diagramZh : problem.diagramEn;
  const category = problem.category || '';

  return (
    <div className="px-7 py-6 space-y-6">
      <div>
        <div className="flex items-center gap-3 mb-2">
          <h1 className="text-[22px] tracking-[-0.02em] font-semibold">{locale === 'zh' ? problem.titleZh : problem.title}</h1>
          <Badge variant={problem.difficulty.toLowerCase() as 'easy' | 'medium' | 'hard'}>
            {problem.difficulty.toUpperCase()}
          </Badge>
        </div>
        {category && (
          <div className="flex items-center gap-1.5 mb-1">
            <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: 'var(--text-3)' }} />
            <span className="mono text-[12px] text-text-3">{category}</span>
          </div>
        )}
        <p className="text-sm text-text-2">{t('implementFn', { fn: problem.functionName })}</p>
      </div>

      {description && (
        <div className="space-y-1">{renderDescription(description)}</div>
      )}

      {/* Theory Section */}
      {theory && (
        <div>
          <button
            onClick={() => setTheoryOpen(!theoryOpen)}
            className="flex items-center gap-2 text-sm text-text-2 hover:text-accent transition-colors"
          >
            <BookOpen className="w-4 h-4" />
            <span>{locale === 'zh' ? '原理解析' : 'Theory'}</span>
            {theoryOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>
          {theoryOpen && (
            <div
              className="mt-2 p-3 px-3.5 rounded-[9px] text-sm text-text-2 leading-relaxed space-y-1"
              style={{
                background: 'color-mix(in oklab, var(--accent) 4%, var(--bg))',
                border: '1px solid var(--accent-line)',
                borderLeft: '3px solid var(--accent)',
              }}
            >
              <span className="mono text-[10.5px] tracking-[0.12em] uppercase text-accent font-semibold block mb-1">
                {locale === 'zh' ? '原理' : 'THEORY'}
              </span>
              <RichText text={theory} />
            </div>
          )}
        </div>
      )}

      {/* Diagram Section */}
      {diagram && (
        <div>
          <button
            onClick={() => setDiagramOpen(!diagramOpen)}
            className="flex items-center gap-2 text-sm text-text-2 hover:text-accent transition-colors"
          >
            <GitBranch className="w-4 h-4" />
            <span>{locale === 'zh' ? '原理图' : 'Diagram'}</span>
            {diagramOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>
          {diagramOpen && (
            <div
              className="mt-2 p-3 px-3.5 rounded-[9px]"
              style={{
                background: 'color-mix(in oklab, var(--accent) 4%, var(--bg))',
                border: '1px solid var(--accent-line)',
                borderLeft: '3px solid var(--accent)',
              }}
            >
              <span className="mono text-[10.5px] tracking-[0.12em] uppercase text-accent font-semibold block mb-1">
                {locale === 'zh' ? '流程图' : 'DIAGRAM'}
              </span>
              {renderMermaid(diagram)}
            </div>
          )}
        </div>
      )}

      {hint && (
        <div>
          <button
            onClick={() => setHintOpen(!hintOpen)}
            className="flex items-center gap-2 text-sm text-text-2 hover:text-accent transition-colors"
          >
            <Lightbulb className="w-4 h-4" />
            <span>{t('hint')}</span>
            {hintOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>
          {hintOpen && (
            <div
              className="mt-2 p-3 px-3.5 rounded-[9px] text-sm text-text-2 leading-relaxed space-y-1"
              style={{
                background: 'color-mix(in oklab, var(--accent) 4%, var(--bg))',
                border: '1px solid var(--accent-line)',
                borderLeft: '3px solid var(--accent)',
              }}
            >
              <span className="mono text-[10.5px] tracking-[0.12em] uppercase text-accent font-semibold block mb-1">HINT</span>
              {hint.split('\n').map((line, i) => (
                <p key={i}>{parseInline(line)}</p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
