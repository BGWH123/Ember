
import Link from '@/routing/Link';
import { CodeEditor } from '@/components/workspace/CodeEditor';
import { RichText } from '@/components/workspace/MathRenderer';
import { useLocale } from '@/context/LocaleContext';
import { useSolution, solutionCode as extractSolutionCode } from '@/lib/problemContext';

interface Cell {
  type: string;
  source: string;
  role: string;
}

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="flex items-center gap-2.5 mono text-xs tracking-[0.12em] uppercase text-text-3 font-medium mt-7 mb-3">
      {children}
      <span className="flex-1 h-px" style={{ background: 'var(--line)' }} />
    </h3>
  );
}

interface SolutionPageProps {
  problemId: string;
}

export function SolutionPageContent({ problemId }: SolutionPageProps) {
  const { t } = useLocale();
  const cells = useSolution(problemId);

  if (cells.length === 0) {
    return <p className="text-sm text-text-3 p-6">{t('noSolution')}</p>;
  }

  const solutionCode = extractSolutionCode(cells);
  const explanations = cells.filter((c) => c.role === 'explanation');
  const demoCode = cells.filter((c) => c.role === 'demo').map((c) => c.source).join('\n\n');

  return (
    <div>
      {/* Breadcrumbs */}
      <div className="mono text-xs text-text-3 flex items-center gap-1.5 mb-5">
        <Link href="/problems" className="hover:text-text transition-colors">{t('problems')}</Link>
        <span className="opacity-60">/</span>
        <Link href={`/problems/${problemId}`} className="hover:text-text transition-colors">{problemId}</Link>
        <span className="opacity-60">/</span>
        <span className="text-text font-medium">{t('solution')}</span>
      </div>

      <div className="grid grid-cols-[minmax(0,1fr)_260px] max-[900px]:grid-cols-1 gap-10 items-start">
        <div>
          <h1 className="text-[clamp(28px,3.5vw,44px)] font-semibold tracking-[-0.028em] leading-[1.08] mb-4">
            {t('solution')}
          </h1>

          {/* Solution code */}
          {solutionCode && (
            <>
              <SectionHeader>Reference Implementation</SectionHeader>
              <div className="rounded-xl overflow-hidden mb-[18px]" style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}>
                <div
                  className="flex items-center gap-2.5 px-3.5 h-[38px] mono text-xs text-text-2"
                  style={{ borderBottom: '1px solid var(--line)', background: 'color-mix(in oklab, var(--text) 2%, var(--bg-elev))' }}
                >
                  <span className="text-text">{problemId}.py</span>
                  <span className="flex-1" />
                </div>
                <div className="h-[400px]">
                  <CodeEditor value={solutionCode} onChange={() => {}} readOnly />
                </div>
              </div>
            </>
          )}

          {/* Explanations */}
          {explanations.length > 0 && (
            <>
              <SectionHeader>Editorial & Key Notes</SectionHeader>
              <div
                className="rounded-xl p-5 mb-[18px] space-y-3"
                style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
              >
                {explanations.map((c, i) => (
                  <RichText key={i} text={c.source} />
                ))}
              </div>
            </>
          )}

          {/* Demo code */}
          {demoCode && (
            <>
              <SectionHeader>Demo</SectionHeader>
              <div className="rounded-xl overflow-hidden" style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}>
                <div
                  className="flex items-center gap-2.5 px-3.5 h-[38px] mono text-xs text-text-2"
                  style={{ borderBottom: '1px solid var(--line)', background: 'color-mix(in oklab, var(--text) 2%, var(--bg-elev))' }}
                >
                  <span className="text-text">demo</span>
                </div>
                <div style={{ height: `${Math.max(120, demoCode.split('\n').length * 22 + 32)}px` }}>
                  <CodeEditor value={demoCode} onChange={() => {}} readOnly />
                </div>
              </div>
            </>
          )}
        </div>

        {/* TOC sidebar */}
        <aside className="sticky top-[76px] hidden md:block">
          <h5 className="eyebrow mb-3">{t('solution')}</h5>
          <div className="flex flex-col gap-1">
            <Link
              href={`/problems/${problemId}`}
              className="text-[13px] text-text-2 hover:text-text transition-colors px-2.5 py-1.5 rounded-[7px] hover:bg-[color-mix(in_oklab,var(--text)_5%,transparent)]"
            >
              ← {t('backToProblem')}
            </Link>
          </div>
        </aside>
      </div>
    </div>
  );
}
