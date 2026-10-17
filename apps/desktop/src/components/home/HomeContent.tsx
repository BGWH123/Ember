import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { ArrowRight, BookOpen, Check, Code2, Cpu, FolderOpen, Route } from 'lucide-react';
import Link from '@/routing/Link';
import { appFetch } from '@/transport/client';
import { useLocale } from '@/context/LocaleContext';
import { Button } from '@/components/ui/Button';
import type { LearningPath, Problem, ProgressMap } from '@/lib/types';

type PathSummary = LearningPath & { solved: number; total: number };

export function HomeContent({ stats }: { stats: { total: number; easy: number; medium: number; hard: number } }) {
  const { locale } = useLocale();
  const zh = locale === 'zh';
  const [problems, setProblems] = useState<Problem[]>([]);
  const [paths, setPaths] = useState<PathSummary[]>([]);
  const [progress, setProgress] = useState<ProgressMap>({});
  const [available, setAvailable] = useState<boolean | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    let alive = true;
    void Promise.all(['/api/problems', '/api/paths', '/api/progress'].map(async path => {
      const response = await appFetch(path);
      if (!response.ok) throw new Error(zh ? '工作区数据暂时无法读取。' : 'Workspace data is unavailable.');
      return response.json();
    })).then(([catalog, routes, records]) => {
      if (!alive) return;
      setProblems(catalog.problems ?? []); setPaths(routes.paths ?? []); setProgress(records.progress ?? {});
    }).catch(error => { if (alive) setError(String(error)); });
    void invoke<{ available: boolean }>('evaluator_status').then(status => { if (alive) setAvailable(status.available); }).catch(() => {});
    return () => { alive = false; };
  }, [zh]);
  const solved = Object.values(progress).filter(item => item.status === 'solved').length;
  const attempted = Object.values(progress).filter(item => item.status === 'attempted').length;
  const last = localStorage.getItem('ember:last-problem');
  const resume = problems.find(problem => problem.id === last) ?? problems.find(problem => problem.id === 'relu');
  const inProgress = problems.filter(problem => progress[problem.id]?.status === 'attempted');
  const exercises = (inProgress.length ? inProgress : ['relu', 'softmax', 'layernorm'].flatMap(id => problems.find(problem => problem.id === id) ?? [])).slice(0, 4);

  return <main className="desktop-dashboard">
    <header className="dashboard-heading"><div><p className="eyebrow">{zh ? '你的学习空间' : 'YOUR SPACE'}</p><h1>{zh ? '工作台' : 'Workspace'}</h1><p>{zh ? '接着上次的思路，继续写下一段代码。' : 'Pick up where you left off. One implementation at a time.'}</p></div><span className="dashboard-device"><span />{zh ? '保存在本机' : 'Saved on this device'}</span></header>
    {error && <p role="alert" className="text-sm text-hard mb-4">{error}</p>}
    <div className="dashboard-overview">
      {[{ label: zh ? '已完成' : 'Completed', value: solved, note: zh ? '道练习' : 'exercises' },
        { label: zh ? '进行中' : 'In progress', value: attempted, note: zh ? '待继续' : 'to revisit' },
        { label: zh ? '题库' : 'Library', value: stats.total || problems.length || '—', note: zh ? '道练习可探索' : 'exercises to explore' }].map(item =>
        <div className="dashboard-stat" key={item.label}><span>{item.label}</span><div><strong>{item.value}</strong><small>{item.note}</small></div></div>)}
    </div>
    <div className="dashboard-primary-grid">
      <section className="dashboard-resume">
        <div className="dashboard-section-label"><Code2 size={15} />{zh ? (last ? '继续练习' : '从这里开始') : (last ? 'CONTINUE PRACTICING' : 'START HERE')}</div>
        <h2>{resume ? (zh ? resume.titleZh : resume.title) : (zh ? '打开你的第一道题' : 'Your first implementation')}</h2>
        <p>{resume ? (zh ? resume.hintZh : resume.hint) : (zh ? '在编辑器中实现一个模块，用真实测试验证你的想法。' : 'Implement a module in the editor and check it with real tests.')}</p>
        <div className="dashboard-resume__footer"><span className="mono">{resume ? `${resume.id}.py` : 'Python · PyTorch'}</span><Link href={resume ? `/problems/${resume.id}` : '/problems'}><Button size="sm">{zh ? (last ? '继续编写' : '打开编辑器') : (last ? 'Continue editing' : 'Open editor')}<ArrowRight size={14} /></Button></Link></div>
      </section>
      <section className="dashboard-runtime"><span className="dashboard-runtime__icon"><Cpu size={20} /></span><h2>{zh ? '本地评测环境' : 'Local evaluator'}</h2><p>{available === null ? (zh ? '正在检查运行环境…' : 'Checking your runtime…') : available ? (zh ? '运行环境已就绪。代码在本机执行，结果与进度保存在工作区。' : 'Ready to run. Code executes locally; results and progress stay in your workspace.') : (zh ? '安装评测模块后，即可运行和提交代码。浏览题目与参考答案始终可用。' : 'Install a module to run and submit code. Exercises and reference solutions are always available.')}</p><Link href="/settings" className="dashboard-text-link">{available ? (zh ? '管理运行环境' : 'Manage runtime') : (zh ? '设置评测环境' : 'Set up evaluator')}<ArrowRight size={13} /></Link></section>
    </div>
    <div className="dashboard-secondary-grid">
      <section className="dashboard-list-section"><div className="dashboard-section-header"><h2>{zh ? (inProgress.length ? '进行中的练习' : '推荐起点') : (inProgress.length ? 'In progress' : 'Good places to start')}</h2><Link href="/problems">{zh ? '全部题目' : 'All exercises'}<ArrowRight size={12} /></Link></div>
        <div className="dashboard-exercise-list">{exercises.map(problem => <Link href={`/problems/${problem.id}`} key={problem.id} className="dashboard-exercise"><span className="dashboard-file-icon">{progress[problem.id]?.status === 'solved' ? <Check size={16} /> : <Code2 size={16} />}</span><span className="dashboard-exercise__name"><strong>{zh ? problem.titleZh : problem.title}</strong><small>{problem.category ?? 'PyTorch'} · {problem.difficulty}</small></span><ArrowRight size={14} /></Link>)}</div>
      </section>
      <section className="dashboard-list-section"><div className="dashboard-section-header"><h2>{zh ? '学习路线' : 'Learning paths'}</h2><Link href="/paths">{zh ? '查看全部' : 'View all'}<ArrowRight size={12} /></Link></div><div className="dashboard-path-list">{paths.slice(0, 3).map(path => <Link href={`/paths/${path.id}`} className="dashboard-path" key={path.id}><div><Route size={15} /><strong>{zh ? path.titleZh : path.titleEn}</strong><small>{path.solved}/{path.total}</small></div><progress aria-label={zh ? path.titleZh : path.titleEn} max={path.total || 1} value={path.solved} /></Link>)}</div></section>
    </div>
    <div className="dashboard-shortcuts"><Link href="/curriculum"><BookOpen size={15} /><span>{zh ? '探索知识体系' : 'Explore the knowledge map'}</span><ArrowRight size={13} /></Link><Link href="/bagu"><FolderOpen size={15} /><span>{zh ? '翻阅理论手册' : 'Open the theory handbook'}</span><ArrowRight size={13} /></Link></div>
  </main>;
}
