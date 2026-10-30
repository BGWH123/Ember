
import { CodeEditor } from './CodeEditor';
import { useLocale } from '@/context/LocaleContext';
import { useSolution, solutionCode as extractSolutionCode } from '@/lib/problemContext';
import { RichText } from './MathRenderer';

interface Cell {
  type: string;
  source: string;
  role: string;
}

interface SolutionTabProps {
  problemId: string;
}

export function SolutionTab({ problemId }: SolutionTabProps) {
  const { t, locale } = useLocale();
  const zh = locale === 'zh';
  const cells = useSolution(problemId);

  if (cells.length === 0) {
    return <div className="p-6 text-sm text-text-3">{t('noSolution')}</div>;
  }

  const solutionCode = extractSolutionCode(cells);
  const demoCode = cells.filter((c) => c.role === 'demo').map((c) => c.source).join('\n\n');
  const explanations = cells.filter((c) => c.role === 'explanation');

  return (
    <div className="h-full overflow-y-auto p-4 space-y-5">
      {solutionCode && (
        <div className="flex flex-col gap-1.5">
          <p className="text-xs font-medium text-text-2 px-1">
            {zh ? '标准参考实现 (Reference Solution)' : 'Reference Solution'}
          </p>
          <div className="rounded-lg overflow-hidden" style={{ border: '1px solid var(--line)' }}>
            <CodeEditor
              value={solutionCode}
              onChange={() => {}}
              readOnly
              height={`${Math.max(120, solutionCode.split('\n').length * 22 + 32)}px`}
              allowParentScrollOnWheel
            />
          </div>
        </div>
      )}
      {explanations.length > 0 && (
        <div className="flex flex-col gap-1.5">
          <p className="text-xs font-medium text-text-2 px-1">
            {zh ? '题解与重点注意分析 (Editorial & Key Notes)' : 'Editorial & Key Notes'}
          </p>
          <div
            className="rounded-lg p-4 space-y-3"
            style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
          >
            {explanations.map((c, i) => (
              <RichText key={i} text={c.source} />
            ))}
          </div>
        </div>
      )}
      {demoCode && (
        <div className="flex flex-col gap-1.5">
          <p className="text-xs font-medium text-text-2 px-1">
            {zh ? '运行验证示例 (Demo)' : 'Interactive Demo'}
          </p>
          <div className="rounded-lg overflow-hidden" style={{ border: '1px solid var(--line)' }}>
            <CodeEditor
              value={demoCode}
              onChange={() => {}}
              readOnly
              height={`${Math.max(80, demoCode.split('\n').length * 22 + 32)}px`}
              allowParentScrollOnWheel
            />
          </div>
        </div>
      )}
    </div>
  );
}
