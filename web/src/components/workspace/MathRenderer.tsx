'use client';

import React from 'react';

/*
  Lightweight LaTeX-to-HTML renderer using pure CSS.
  Supports: fractions, sub/superscripts, Greek letters, roots, sums/integrals,
  arrows, common operators, and text blocks.
  No external dependencies — works without npm install.
*/

const GREEK: Record<string, string> = {
  alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ε', zeta: 'ζ',
  eta: 'η', theta: 'θ', iota: 'ι', kappa: 'κ', lambda: 'λ', mu: 'μ',
  nu: 'ν', xi: 'ξ', omicron: 'ο', pi: 'π', rho: 'ρ', sigma: 'σ', tau: 'τ',
  upsilon: 'υ', phi: 'φ', chi: 'χ', psi: 'ψ', omega: 'ω',
  Gamma: 'Γ', Delta: 'Δ', Theta: 'Θ', Lambda: 'Λ', Xi: 'Ξ', Pi: 'Π',
  Sigma: 'Σ', Phi: 'Φ', Psi: 'Ψ', Omega: 'Ω',
};

const SYMBOLS: Record<string, string> = {
  '\\cdot': '·', '\\odot': '⊙', '\\times': '×', '\\div': '÷',
  '\\pm': '±', '\\mp': '∓', '\\leq': '≤', '\\geq': '≥',
  '\\neq': '≠', '\\approx': '≈', '\\equiv': '≡', '\\sim': '∼',
  '\\rightarrow': '→', '\\leftarrow': '←', '\\Rightarrow': '⇒',
  '\\Leftarrow': '⇐', '\\infty': '∞', '\\partial': '∂',
  '\\nabla': '∇', '\\in': '∈', '\\notin': '∉', '\\forall': '∀',
  '\\exists': '∃', '\\subset': '⊂', '\\subseteq': '⊆',
  '\\cup': '∪', '\\cap': '∩', '\\emptyset': '∅',
  '\\sum': '∑', '\\prod': '∏', '\\int': '∫', '\\oint': '∮',
  '\\sqrt': '√', '\\propto': '∝', '\\angle': '∠',
  '\\perp': '⊥', '\\parallel': '∥', '\\ldots': '…', '\\cdots': '⋯',
  '\\vdots': '⋮', '\\ddots': '⋱',
  '\\mathbb{R}': 'ℝ', '\\mathbb{N}': 'ℕ', '\\mathbb{Z}': 'ℤ',
  '\\mathbb{Q}': 'ℚ', '\\mathbb{C}': 'ℂ',
  '\\mathcal{L}': '𝓛', '\\mathcal{N}': '𝓝', '\\mathcal{D}': '𝒟',
  '\\ell': 'ℓ', '\\hbar': 'ℏ',
};

function replaceSymbols(text: string): string {
  let s = text;
  // Replace \command symbols
  for (const [cmd, ch] of Object.entries(SYMBOLS)) {
    s = s.split(cmd).join(ch);
  }
  // Greek letters: \alpha, \beta, etc.
  s = s.replace(/\\([a-zA-Z]+)(?![a-zA-Z])/g, (m, name) => {
    return GREEK[name] ?? m;
  });
  return s;
}

function parseBraces(text: string, start: number): { content: string; end: number } {
  let depth = 1;
  let i = start + 1;
  while (i < text.length && depth > 0) {
    if (text[i] === '{') depth++;
    else if (text[i] === '}') depth--;
    i++;
  }
  return { content: text.slice(start + 1, i - 1), end: i };
}

function parseLatexToReact(text: string, keyPrefix = 'm'): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  let i = 0;
  let key = 0;

  const pushText = (t: string) => {
    if (!t) return;
    const processed = replaceSymbols(t);
    parts.push(<span key={`${keyPrefix}-${key++}`}>{processed}</span>);
  };

  while (i < text.length) {
    // Fraction \frac{num}{den}
    if (text.slice(i).startsWith('\\frac{')) {
      const numParse = parseBraces(text, i + 5);
      // Find the opening brace for denominator
      const denStart = numParse.end;
      if (denStart < text.length && text[denStart] === '{') {
        const denParse = parseBraces(text, denStart);
        parts.push(
          <span key={`${keyPrefix}-${key++}`} className="inline-flex flex-col items-center align-middle mx-0.5 text-[0.92em]">
            <span className="px-0.5">{parseLatexToReact(numParse.content, `${keyPrefix}-n`)}</span>
            <span className="w-full h-px bg-current opacity-60" />
            <span className="px-0.5">{parseLatexToReact(denParse.content, `${keyPrefix}-d`)}</span>
          </span>
        );
        i = denParse.end;
        continue;
      }
    }

    // Square root \sqrt{...}
    if (text.slice(i).startsWith('\\sqrt{')) {
      const inner = parseBraces(text, i + 5);
      parts.push(
        <span key={`${keyPrefix}-${key++}`} className="inline-flex items-start mx-0.5">
          <span className="text-[1.1em] leading-none mt-0.5">√</span>
          <span className="border-t border-current pt-px px-0.5">{parseLatexToReact(inner.content, `${keyPrefix}-s`)}</span>
        </span>
      );
      i = inner.end;
      continue;
    }

    // Text block \text{...}
    if (text.slice(i).startsWith('\\text{')) {
      const inner = parseBraces(text, i + 5);
      parts.push(<span key={`${keyPrefix}-${key++}`}>{inner.content}</span>);
      i = inner.end;
      continue;
    }

    // Subscript _{...} or _x
    if (text[i] === '_' && i + 1 < text.length) {
      if (text[i + 1] === '{') {
        const inner = parseBraces(text, i + 1);
        parts.push(
          <sub key={`${keyPrefix}-${key++}`} className="text-[0.72em] align-sub leading-none">
            {parseLatexToReact(inner.content, `${keyPrefix}-sub`)}
          </sub>
        );
        i = inner.end;
      } else {
        parts.push(<sub key={`${keyPrefix}-${key++}`} className="text-[0.72em] align-sub leading-none">{text[i + 1]}</sub>);
        i += 2;
      }
      continue;
    }

    // Superscript ^{...} or ^x
    if (text[i] === '^' && i + 1 < text.length) {
      if (text[i + 1] === '{') {
        const inner = parseBraces(text, i + 1);
        parts.push(
          <sup key={`${keyPrefix}-${key++}`} className="text-[0.72em] align-super leading-none">
            {parseLatexToReact(inner.content, `${keyPrefix}-sup`)}
          </sup>
        );
        i = inner.end;
      } else {
        parts.push(<sup key={`${keyPrefix}-${key++}`} className="text-[0.72em] align-super leading-none">{text[i + 1]}</sup>);
        i += 2;
      }
      continue;
    }

    // Left/right parentheses \left( ... \right)
    if (text.slice(i).startsWith('\\left(') || text.slice(i).startsWith('\\left[') || text.slice(i).startsWith('\\left{')) {
      const open = text[i + 5];
      const closeMap: Record<string, string> = { '(': ')', '[': ']', '{': '}' };
      const closeExpected = '\\right' + closeMap[open];
      const endIdx = text.indexOf(closeExpected, i);
      if (endIdx > 0) {
        const inner = text.slice(i + 6, endIdx);
        parts.push(
          <span key={`${keyPrefix}-${key++}`} className="inline-flex items-center mx-px">
            <span className="text-[1.15em] mx-px">{open === '{' ? '{' : open}</span>
            <span>{parseLatexToReact(inner, `${keyPrefix}-p`)}</span>
            <span className="text-[1.15em] mx-px">{open === '{' ? '}' : closeMap[open]}</span>
          </span>
        );
        i = endIdx + closeExpected.length;
        continue;
      }
    }

    // Backslash command (catch-all for remaining \...)
    if (text[i] === '\\') {
      // Try to match a command name
      const match = text.slice(i + 1).match(/^([a-zA-Z]+)/);
      if (match) {
        const cmd = '\\' + match[1];
        const replaced = SYMBOLS[cmd] ?? GREEK[match[1]];
        if (replaced) {
          parts.push(<span key={`${keyPrefix}-${key++}`}>{replaced}</span>);
          i += cmd.length;
          continue;
        }
      }
    }

    // Regular character — accumulate
    let j = i;
    while (j < text.length && text[j] !== '\\' && text[j] !== '_' && text[j] !== '^' && text[j] !== '{') {
      j++;
    }
    pushText(text.slice(i, j));
    i = j;
  }

  return parts;
}

export function InlineMath({ latex }: { latex: string }) {
  return (
    <span className="font-serif text-[0.98em] tracking-tight mx-0.5" style={{ fontFamily: 'Georgia, "Times New Roman", serif' }}>
      {parseLatexToReact(latex)}
    </span>
  );
}

export function DisplayMath({ latex }: { latex: string }) {
  return (
    <div className="my-3 px-4 py-3 rounded-lg text-center overflow-x-auto" style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}>
      <span className="font-serif text-[1.05em] tracking-tight inline-block" style={{ fontFamily: 'Georgia, "Times New Roman", serif' }}>
        {parseLatexToReact(latex)}
      </span>
    </div>
  );
}

/**
 * Parse a markdown-like text string, handling:
 * - **bold**
 * - `code`
 * - $$...$$ display math
 * - $...$ inline math
 */
export function RichText({ text }: { text: string }) {
  const lines = text.split('\n');
  return (
    <div className="text-sm text-text-2 leading-relaxed space-y-1">
      {lines.map((line, idx) => {
        if (line === '') return <div key={idx} className="h-1.5" />;
        if (line.startsWith('**') && line.endsWith('**')) {
          return <strong key={idx} className="font-medium text-text block mt-2.5">{line.slice(2, -2)}</strong>;
        }
        if (line.startsWith('$$') && line.endsWith('$$')) {
          return <DisplayMath key={idx} latex={line.slice(2, -2).trim()} />;
        }
        return <p key={idx}>{parseInline(line)}</p>;
      })}
    </div>
  );
}

function parseInline(text: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  let remaining = text;
  let key = 0;

  while (remaining.length > 0) {
    const boldMatch = remaining.match(/\*\*(.+?)\*\*/);
    const codeMatch = remaining.match(/`(.+?)`/);
    const mathMatch = remaining.match(/\$(.+?)\$/);

    const boldIdx = boldMatch?.index ?? Infinity;
    const codeIdx = codeMatch?.index ?? Infinity;
    const mathIdx = mathMatch?.index ?? Infinity;

    if (boldIdx === Infinity && codeIdx === Infinity && mathIdx === Infinity) {
      parts.push(remaining);
      break;
    }

    const firstIdx = Math.min(boldIdx, codeIdx, mathIdx);

    if (firstIdx > 0) parts.push(remaining.slice(0, firstIdx));

    if (boldIdx === firstIdx && boldMatch) {
      parts.push(<strong key={key++} className="font-medium text-text">{boldMatch[1]}</strong>);
      remaining = remaining.slice(boldIdx + boldMatch[0].length);
    } else if (codeIdx === firstIdx && codeMatch) {
      parts.push(
        <code key={key++} className="mono text-[12px] px-[5px] py-px rounded text-text" style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}>
          {codeMatch[1]}
        </code>
      );
      remaining = remaining.slice(codeIdx + codeMatch[0].length);
    } else if (mathIdx === firstIdx && mathMatch) {
      parts.push(<InlineMath key={key++} latex={mathMatch[1]} />);
      remaining = remaining.slice(mathIdx + mathMatch[0].length);
    } else {
      parts.push(remaining);
      break;
    }
  }
  return parts;
}
