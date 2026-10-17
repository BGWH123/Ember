import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { NavLink, useLocation, useNavigate, useNavigationType } from 'react-router-dom';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { ArrowLeft, ArrowRight, BookOpen, Code2, Copy, Flame, LayoutDashboard, Library,
  MessagesSquare, Minus, Moon, NotebookText, PanelLeft, Route, Settings, Square, Sun, Target, X } from 'lucide-react';
import { useLocale } from '@/context/LocaleContext';
import { useTheme } from '@/context/ThemeContext';

export function DesktopShell({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  const navigationType = useNavigationType();
  const navigate = useNavigate();
  const { locale, setLocale } = useLocale();
  const { theme, toggleTheme } = useTheme();
  const zh = locale === 'zh';
  const [collapsed, setCollapsed] = useState(() => {
    const saved = localStorage.getItem('ember:sidebar-collapsed');
    return saved === null ? window.innerWidth < 1100 : saved === 'true';
  });
  const [maximized, setMaximized] = useState(false);
  const [windowError, setWindowError] = useState('');
  const [nativeWindow] = useState(() => { try { return getCurrentWindow(); } catch { return null; } });
  const scroll = useRef<HTMLDivElement>(null);
  const positions = useRef(new Map<string, number>());
  const historyIndex = Number(window.history.state?.idx ?? 0);
  const furthest = useRef(historyIndex);
  if (navigationType === 'POP') furthest.current = Math.max(furthest.current, historyIndex);
  else furthest.current = historyIndex;

  const groups = [
    { label: zh ? '学习空间' : 'LEARN', items: [
      { path: '/', label: zh ? '工作台' : 'Workspace', icon: LayoutDashboard },
      { path: '/problems', label: zh ? '题库' : 'Problems', icon: Library },
      { path: '/paths', label: zh ? '学习路线' : 'Learning paths', icon: Route },
      { path: '/curriculum', label: zh ? '知识体系' : 'Knowledge map', icon: BookOpen },
    ] },
    { label: zh ? '面试准备' : 'PREPARE', items: [
      { path: '/hot', label: zh ? '热题 50' : 'Hot 50', icon: Target },
      { path: '/bagu', label: zh ? '理论手册' : 'Theory handbook', icon: NotebookText },
      { path: '/interview', label: zh ? '模拟面试' : 'Mock interview', icon: MessagesSquare },
    ] },
  ];
  const workspace = /^\/problems\/[^/]+$/.test(pathname);
  const current = groups.flatMap(group => group.items).find(item => item.path === '/' ? pathname === '/' : pathname.startsWith(item.path));
  const title = workspace ? `${decodeURIComponent(pathname.split('/').at(-1) ?? '')}.py`
    : pathname.startsWith('/settings') ? (zh ? '设置' : 'Settings')
    : pathname.startsWith('/solutions') ? (zh ? '参考答案' : 'Reference solution') : current?.label ?? 'Ember';

  useEffect(() => {
    localStorage.setItem('ember:sidebar-collapsed', String(collapsed));
  }, [collapsed]);
  useEffect(() => {
    const compact = window.matchMedia('(max-width: 1099px)');
    const fit = (event: MediaQueryListEvent) => { if (event.matches) setCollapsed(true); };
    compact.addEventListener('change', fit);
    return () => compact.removeEventListener('change', fit);
  }, []);
  useEffect(() => {
    document.title = `${title} — Ember`;
    if (workspace) localStorage.setItem('ember:last-problem', pathname.split('/').at(-1)!);
  }, [pathname, title, workspace]);
  useLayoutEffect(() => {
    const element = scroll.current;
    if (!element) return;
    element.scrollTop = positions.current.get(pathname) ?? 0;
    return () => { positions.current.set(pathname, element.scrollTop); };
  }, [pathname]);
  useEffect(() => {
    function shortcut(event: KeyboardEvent) {
      if (!(event.ctrlKey || event.metaKey) || event.altKey) return;
      if (event.key === ',') { event.preventDefault(); navigate('/settings'); }
      if (event.key.toLowerCase() === 'b' && !(event.target as HTMLElement).closest('input, textarea, [contenteditable="true"], .monaco-editor')) {
        event.preventDefault(); setCollapsed(value => !value);
      }
    }
    window.addEventListener('keydown', shortcut);
    return () => window.removeEventListener('keydown', shortcut);
  }, [navigate]);
  useEffect(() => {
    if (!nativeWindow) return;
    let disposed = false;
    let unlisten: (() => void) | undefined;
    const refresh = async () => { const value = await nativeWindow.isMaximized(); if (!disposed) setMaximized(value); };
    void (async () => {
      await refresh();
      const stop = await nativeWindow.onResized(() => { void refresh().catch(console.error); });
      if (disposed) stop(); else unlisten = stop;
    })().catch(console.error);
    return () => { disposed = true; unlisten?.(); };
  }, [nativeWindow]);
  async function windowAction(action: 'minimize' | 'toggleMaximize' | 'close') {
    if (!nativeWindow) return;
    try { await nativeWindow[action](); setWindowError(''); }
    catch { setWindowError(zh ? '窗口操作失败，请重试。' : 'Window action failed. Please try again.'); }
  }

  return <div className={`desktop-shell${collapsed ? ' desktop-shell--collapsed' : ''}${maximized ? ' desktop-shell--maximized' : ''}`}>
    <header className="desktop-titlebar" data-tauri-drag-region>
      <div className="desktop-brand" data-tauri-drag-region><span className="desktop-brand__mark"><Flame size={15} /></span><span>Ember</span></div>
      <div className="desktop-titlebar__tools">
        <button className="desktop-icon-button" aria-label={zh ? '切换侧边栏' : 'Toggle sidebar'} aria-expanded={!collapsed} title={zh ? '切换侧边栏 · Ctrl+B' : 'Toggle sidebar · Ctrl+B'} onClick={() => setCollapsed(value => !value)}><PanelLeft size={16} /></button>
        <span className="desktop-titlebar__separator" />
        <button className="desktop-icon-button" aria-label={zh ? '后退' : 'Back'} disabled={historyIndex === 0} onClick={() => navigate(-1)}><ArrowLeft size={15} /></button>
        <button className="desktop-icon-button" aria-label={zh ? '前进' : 'Forward'} disabled={historyIndex >= furthest.current} onClick={() => navigate(1)}><ArrowRight size={15} /></button>
      </div>
      <div className="desktop-titlebar__context" data-tauri-drag-region>{workspace && <Code2 size={13} />}<span>{title}</span></div>
      <div className="desktop-titlebar__preferences">
        <button className="desktop-icon-button" aria-label={zh ? '切换主题' : 'Toggle theme'} onClick={toggleTheme}>{theme === 'light' ? <Sun size={15} /> : <Moon size={15} />}</button>
        <button className="desktop-locale-button" aria-label={zh ? '切换语言' : 'Change language'} onClick={() => setLocale(zh ? 'en' : 'zh')}>{zh ? '中文' : 'EN'}</button>
      </div>
      <div className="desktop-window-controls">
        <button aria-label={zh ? '最小化' : 'Minimize'} disabled={!nativeWindow} onClick={() => void windowAction('minimize')}><Minus size={13} /></button>
        <button aria-label={zh ? (maximized ? '还原窗口' : '最大化') : (maximized ? 'Restore' : 'Maximize')} disabled={!nativeWindow} onClick={() => void windowAction('toggleMaximize')}>{maximized ? <Copy size={11} /> : <Square size={11} />}</button>
        <button className="desktop-window-close" aria-label={zh ? '关闭窗口' : 'Close window'} disabled={!nativeWindow} onClick={() => void windowAction('close')}><X size={15} /></button>
      </div>
    </header>
    <div className="desktop-body">
      <aside className="desktop-sidebar" aria-label={zh ? '应用导航' : 'App navigation'}>
        <div className="desktop-sidebar__navigation">{groups.map(group => <div className="desktop-nav-group" key={group.label}>
          <p className="desktop-nav-group__label">{group.label}</p>
          <nav>{group.items.map(({ path, label, icon: Icon }) => <NavLink key={path} to={path} end={path === '/'} title={collapsed ? label : undefined} aria-label={label} className={({ isActive }) => `desktop-nav-item${isActive ? ' desktop-nav-item--active' : ''}`}><Icon size={17} /><span>{label}</span></NavLink>)}</nav>
        </div>)}</div>
        <div className="desktop-sidebar__footer">
          <NavLink to="/settings" aria-label={zh ? '设置' : 'Settings'} title={collapsed ? (zh ? '设置' : 'Settings') : undefined} className={({ isActive }) => `desktop-nav-item${isActive ? ' desktop-nav-item--active' : ''}`}><Settings size={17} /><span>{zh ? '设置' : 'Settings'}</span><kbd>⌃ ,</kbd></NavLink>
          <div className="desktop-local-indicator" title={zh ? '学习记录保存在本机' : 'Learning records stay on this device'}><span className="desktop-local-indicator__dot" /><span>{zh ? '本地工作区' : 'Local workspace'}</span></div>
        </div>
      </aside>
      <div ref={scroll} data-desktop-scroll className={`desktop-content${workspace ? ' desktop-content--workspace' : ''}`}>
        {windowError && <p role="alert" className="desktop-window-error">{windowError}</p>}
        {children}
      </div>
    </div>
  </div>;
}
