import { invoke } from '@tauri-apps/api/core';
import { appFetch } from '@/transport/client';

import { useEffect, useState, useRef } from 'react';
import { useParams, useSearchParams, useRouter } from '@/routing/navigation';
import Link from '@/routing/Link';
import * as Tabs from '@radix-ui/react-tabs';
import { Menu, ChevronLeft, ChevronRight, RotateCcw } from 'lucide-react';
import { SplitPane, VerticalSplitPane } from '@/components/ui/SplitPane';
import { ProblemDrawer } from '@/components/layout/ProblemDrawer';
import { FeedbackModal } from '@/components/workspace/FeedbackModal';
import { DescriptionTab } from '@/components/workspace/DescriptionTab';
import { SolutionTab } from '@/components/workspace/SolutionTab';
import { AIHelpTab } from '@/components/workspace/AIHelpTab';
import { CodeEditor } from '@/components/workspace/CodeEditor';
import { TestPanel } from '@/components/workspace/TestPanel';
import { ActionBar } from '@/components/workspace/ActionBar';
import { WaitingState } from '@/components/ui/WaitingState';
import { useProblemStore } from '@/store/problemStore';
import { useLocale } from '@/context/LocaleContext';
import type { Problem, ProgressMap, SubmissionResult, LearningPath, LearningPathProblemSummary, SubmissionHistory } from '@/lib/types';

export default function WorkspacePage() {
  const { id } = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathId = searchParams.get('path');
  const { locale, t } = useLocale();
  const {
    currentCode, setCurrentCode,
    submissionResult, setSubmissionResult,
    isSubmitting, setIsSubmitting,
    drawerOpen, setDrawerOpen,
    isRunning, setIsRunning,
    setRunResult, setBottomTab, resetTestPanel, resetAiHelp,
    submissionHistory, setSubmissionHistory,
  } = useProblemStore();

  const [loadError,setLoadError] = useState('');
  const [draftReady, setDraftReady] = useState(false);
  const [problem, setProblem] = useState<(Problem & { starterCode?: string }) | null>(null);
  const [allProblems, setAllProblems] = useState<Problem[]>([]);
  const [progress, setProgress] = useState<ProgressMap>({});
  const [pathData, setPathData] = useState<(Omit<LearningPath, 'problems'> & { problems: LearningPathProblemSummary[] }) | null>(null);
  const [feedbackResult, setFeedbackResult] = useState<SubmissionResult | null>(null);


  const draftSnapshot = useRef<{id:string;code:string}|null>(null);
  const readyId = useRef<string | null>(null);
  if (draftReady && readyId.current === id) draftSnapshot.current = {id,code:currentCode};
  useEffect(() => {
    let alive = true;
    readyId.current = null;
    setDraftReady(false);
    setProblem(null);
    appFetch(`/api/problems/${id}`).then(async response => {
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Problem unavailable');
      const draft = await invoke<string | null>('load_draft', { id });
      if (!alive) return;
      readyId.current = id;
      setCurrentCode(draft ?? data.starterCode ?? '');
      setProblem(data);
      setDraftReady(true);
      setSubmissionResult(null);
      resetTestPanel();
      resetAiHelp();
    }).catch(error => { if (alive) setLoadError(String(error)); });
    appFetch('/api/problems').then(r=>r.json()).then(data=>{if(alive)setAllProblems(data.problems || []);});
    appFetch('/api/progress').then(r=>r.json()).then(data=>{if(alive)setProgress(data.progress || {});});
    appFetch(`/api/submissions/${id}`).then(r=>r.json()).then(data=>{if(alive&&Array.isArray(data))setSubmissionHistory(data);});
    return () => {
      alive=false;
      const draft=draftSnapshot.current;
      if(draft) void invoke('save_draft',draft).catch(console.error);
    };
  }, [id, setCurrentCode, setSubmissionResult, resetTestPanel, resetAiHelp, setSubmissionHistory]);
  useEffect(()=>{
    let alive=true;
    if(pathId) appFetch(`/api/paths/${pathId}`).then(r=>r.json()).then(data=>{if(alive)setPathData(data);});
    else setPathData(null);
    return ()=>{alive=false;};
  },[pathId]);
  useEffect(() => {
    if (!draftReady || readyId.current !== id) return;
    const timer = setTimeout(() => { void invoke('save_draft', { id, code: currentCode }).catch(console.error); }, 500);
    return () => clearTimeout(timer);
  }, [id, currentCode, draftReady]);

  const handleRestore = () => {
    const starter = problem?.starterCode;
    if (!starter || starter === currentCode || readyId.current !== id) return;
    setCurrentCode(starter);
    draftSnapshot.current = {id, code: starter};
    void invoke('save_draft', { id, code: starter }).catch(console.error);
  };

  const handleRun = async () => {
    if (!problem || isRunning || isSubmitting) return;
    setIsRunning(true);
    setBottomTab('testresults');
    setRunResult(null);
    try {
      const testIndices = problem.tests.slice(0, 2).map((_, i) => i);
      const res = await appFetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ taskId: id, code: currentCode, testIndices }),
      });
      const accepted = await res.json();
      if (!res.ok || !accepted.submission_id) throw new Error(accepted.detail?.message || accepted.message || t('networkError'));
      const data = accepted as SubmissionResult;
      setRunResult(data);
      setBottomTab('testresults');
    } catch (error) {
      setRunResult({ passed: 0, total: 0, allPassed: false, results: [], totalTimeMs: 0, error: error instanceof Error ? error.message : t('networkError') });
    } finally {
      setIsRunning(false);
    }
  };

  const handleSubmit = async () => {
    if (!problem || isSubmitting || isRunning) return;
    setIsSubmitting(true);
    setBottomTab('testresults');
    setSubmissionResult(null);
    try {
      const res = await appFetch('/api/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ taskId: id, code: currentCode }),
      });
      const accepted = await res.json();
      if (!res.ok || !accepted.submission_id) throw new Error(accepted.detail?.message || accepted.message || t('networkError'));
      const data = accepted as SubmissionResult;
      setSubmissionResult(data);
      setRunResult(data);
      setBottomTab('testresults');
      setFeedbackResult(data);
      appFetch('/api/progress').then((r) => r.json()).then((d) => setProgress(d.progress || {}));
      appFetch(`/api/submissions/${id}`).then((r) => { if (!r.ok) throw new Error('History unavailable'); return r.json(); }).then((d: SubmissionHistory[]) => setSubmissionHistory(d)).catch(() => {});
    } catch (error) {
      const err = { passed: 0, total: 0, allPassed: false, results: [], totalTimeMs: 0, error: error instanceof Error ? error.message : t('networkError') };
      setSubmissionResult(err);
      setRunResult(err);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (loadError) return <main className="p-8" role="alert">{loadError} · <Link href="/problems">返回题库</Link></main>;
  if (!problem) {
    return (
      <WaitingState />
    );
  }

  // Path navigation
  const pathProblems = pathData?.problems ?? [];
  const currentPathIdx = pathProblems.findIndex((p) => p.id === id);
  const prevProblem = currentPathIdx > 0 ? pathProblems[currentPathIdx - 1] : null;
  const nextProblem = currentPathIdx >= 0 && currentPathIdx < pathProblems.length - 1 ? pathProblems[currentPathIdx + 1] : null;

  const leftPanel = (
    <div className="h-full flex flex-col bg-bg">
      <Tabs.Root defaultValue="description" className="flex flex-col h-full">
        <div
          className="flex items-center px-3.5 h-10 gap-1 flex-shrink-0"
          style={{ borderBottom: '1px solid var(--line)', background: 'var(--bg)' }}
        >
          <Tabs.List className="flex gap-0.5">
            {(['description', 'solution', 'aiHelp'] as const).map((tab) => (
              <Tabs.Trigger
                key={tab}
                value={tab}
                className="group px-3 h-[26px] rounded-[6px] text-[13px] text-text-2 cursor-pointer inline-flex items-center gap-1.5 transition-[background,color] duration-150 data-[state=active]:bg-[color-mix(in_oklab,var(--text)_7%,transparent)] data-[state=active]:text-text"
              >
                <span className="w-[5px] h-[5px] rounded-full bg-accent hidden group-data-[state=active]:inline-block" />
                {t(tab === 'description' ? 'description' : tab === 'solution' ? 'solution' : 'aiHelp')}
              </Tabs.Trigger>
            ))}
          </Tabs.List>
        </div>
        <Tabs.Content value="description" className="flex-1 overflow-y-auto">
          <DescriptionTab problem={problem} />
        </Tabs.Content>
        <Tabs.Content value="solution" className="flex-1 overflow-y-auto">
          <SolutionTab problemId={id} />
        </Tabs.Content>
        <Tabs.Content value="aiHelp" className="flex-1 overflow-y-auto">
          <AIHelpTab problem={problem} />
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );

  const rightPanel = (
    <div className="h-full flex flex-col">
      <VerticalSplitPane
        top={
          <div className="flex flex-col h-full">
            <div
              className="flex items-center gap-2.5 px-3.5 flex-shrink-0"
              style={{
                height: '36px',
                borderBottom: '1px solid var(--line)',
                background: 'color-mix(in oklab, var(--text) 2%, var(--bg))',
              }}
            >
              <svg className="w-3.5 h-3.5 text-text-3" viewBox="0 0 16 16" fill="none">
                <path d="M3 1.5h7l3 3V14.5H3V1.5z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
                <path d="M10 1.5v3h3" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
              </svg>
              <span className="mono text-[12.5px] text-text-2">{id}.py</span>
              <button
                type="button"
                onClick={handleRestore}
                disabled={!problem.starterCode || currentCode === problem.starterCode}
                className="ml-auto inline-flex items-center gap-1.5 h-6 px-2 rounded text-[11.5px] text-text-2 cursor-pointer hover:text-text disabled:opacity-40 disabled:pointer-events-none"
                style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}
              >
                <RotateCcw className="w-3 h-3" />
                {t('restoreTemplate')}
              </button>
              <span className="mono text-[11.5px] text-text-3 px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}>Python</span>
            </div>
            <div className="flex-1 overflow-hidden">
              <CodeEditor value={currentCode} onChange={setCurrentCode} />
            </div>
          </div>
        }
        bottom={
          <div className="flex flex-col h-full">
            <TestPanel tests={problem.tests} functionName={problem.functionName} />
            <ActionBar onSubmit={handleSubmit} onRun={handleRun} isSubmitting={isSubmitting} isRunning={isRunning} attemptCount={submissionHistory.length} />
          </div>
        }
        defaultRatio={0.65}
        minTop={200}
        minBottom={150}
      />
    </div>
  );

  return (
    <div className="workspace-document bg-bg">
      <div className="workspace-toolbar">
        {/* Breadcrumbs */}
        <div className="text-xs text-text-3 flex items-center gap-1.5 min-w-0">
          <Link href="/problems" className="text-text-2 hover:text-text transition-colors">{t('problems')}</Link>
          <span className="opacity-60">/</span>
          <span className="text-text font-medium">{problem ? (locale === 'zh' ? problem.titleZh : problem.title) : id}</span>
        </div>

        <button
          aria-label={locale === 'zh' ? '打开题目列表' : 'Open problem list'}
          onClick={() => setDrawerOpen(true)}
          className="w-8 h-8 inline-flex items-center justify-center rounded-lg text-text-2 cursor-pointer hover:text-text hover:border-line-strong transition-[color,border-color] duration-150"
          style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
        >
          <Menu className="w-3.5 h-3.5" />
        </button>

        <div className="flex-1" />

        {/* Path progress */}
        {pathData && pathProblems.length > 0 && (
          <div
            className="flex items-center gap-2.5 px-2.5 py-1 rounded-pill mono text-xs text-text-2"
            style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
          >
            <div className="flex gap-[3px]">
              {pathProblems.map((p, i) => (
                <span
                  key={i}
                  className="w-1.5 h-1.5 rounded-full"
                  style={{
                    background:
                      p.id === id
                        ? 'var(--accent)'
                        : p.status === 'solved'
                        ? 'var(--easy)'
                        : p.status === 'attempted'
                        ? 'var(--medium)'
                        : 'var(--line-strong)',
                    boxShadow: p.id === id ? '0 0 0 2px color-mix(in oklab, var(--accent) 25%, transparent)' : undefined,
                  }}
                />
              ))}
            </div>
            <span>{currentPathIdx + 1}/{pathProblems.length}</span>
          </div>
        )}

        {/* Nav arrows */}
        {pathData && (
          <div className="flex gap-1">
            <button
              onClick={() => prevProblem && router.push(`/problems/${prevProblem.id}?path=${pathId}`)}
              disabled={!prevProblem}
              className="w-7 h-7 inline-flex items-center justify-center rounded-[7px] text-text-2 cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed hover:text-text hover:border-line-strong transition-[color,border-color] duration-150"
              style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => nextProblem && router.push(`/problems/${nextProblem.id}?path=${pathId}`)}
              disabled={!nextProblem}
              className="w-7 h-7 inline-flex items-center justify-center rounded-[7px] text-text-2 cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed hover:text-text hover:border-line-strong transition-[color,border-color] duration-150"
              style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

      </div>

      <div className="flex-1 overflow-hidden" style={{ background: 'var(--bg-sunken)' }}>
        <SplitPane left={leftPanel} right={rightPanel} />
      </div>
      <ProblemDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        problems={allProblems}
        progress={progress}
        currentId={id}
      />
      {feedbackResult && (
        <FeedbackModal
          result={feedbackResult}
          onClose={() => setFeedbackResult(null)}
          attemptNumber={submissionHistory.length}
          pathProgress={pathData ? {
            solved: pathData.problems.filter(p => p.status === 'solved').length,
            total: pathData.problems.length,
            name: locale === 'zh' ? pathData.titleZh : pathData.titleEn,
          } : null}
        />
      )}
    </div>
  );
}
