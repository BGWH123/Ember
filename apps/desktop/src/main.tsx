import React,{lazy,Suspense} from 'react';
import {createRoot} from 'react-dom/client';
import {HashRouter,Routes,Route} from 'react-router-dom';
import {LocaleProvider} from '@/context/LocaleContext';
import {ThemeProvider} from '@/context/ThemeContext';
import {ThemeScript} from '@/components/ThemeScript';
import {WaitingState} from '@/components/ui/WaitingState';
import {DesktopShell} from '@/components/layout/DesktopShell';
import '@fontsource-variable/noto-sans';
import '@fontsource-variable/noto-sans/wght-italic.css';
import '@fontsource-variable/noto-sans-sc';
import '@fontsource-variable/fira-code';
import './app/globals.css';
const Home=lazy(()=>import('./app/page'));
const Problems=lazy(()=>import('./app/problems/page'));
const Problem=lazy(()=>import('./app/problems/[id]/page'));
const Paths=lazy(()=>import('./app/paths/page'));
const Path=lazy(()=>import('./app/paths/[id]/page'));
const Solution=lazy(()=>import('./app/solutions/[id]/page'));
const Curriculum=lazy(()=>import('./app/curriculum/page'));
const Bagu=lazy(()=>import('./app/bagu/page'));
const BaguDetail=lazy(()=>import('./app/bagu/[slug]/page'));
const Hot=lazy(()=>import('./app/hot/page'));
const Interview=lazy(()=>import('./app/interview/page'));
const Settings=lazy(()=>import('./settings'));
function WorkspaceRoute(){return <Problem/>;}
class ErrorBoundary extends React.Component<{children:React.ReactNode},{error:string}>{
  state={error:''};static getDerivedStateFromError(error:Error){return{error:error.message};}
  render(){return this.state.error?<main className="p-8"><h1>页面无法加载</h1><p>{this.state.error}</p><a href="#/settings">打开设置</a></main>:this.props.children;}
}
createRoot(document.getElementById('root')!).render(<ErrorBoundary><ThemeScript/><ThemeProvider><LocaleProvider><HashRouter><DesktopShell><Suspense fallback={<WaitingState/>}><Routes>
  <Route path="/" element={<Home/>}/><Route path="/problems" element={<Problems/>}/><Route path="/problems/:id" element={<WorkspaceRoute/>}/><Route path="/paths" element={<Paths/>}/><Route path="/paths/:id" element={<Path/>}/><Route path="/solutions/:id" element={<Solution/>}/><Route path="/curriculum" element={<Curriculum/>}/><Route path="/bagu" element={<Bagu/>}/><Route path="/bagu/:kind/:slug" element={<BaguDetail/>}/><Route path="/hot" element={<Hot/>}/><Route path="/interview" element={<Interview/>}/><Route path="/settings" element={<Settings/>}/><Route path="*" element={<p className="p-8">页面不存在 · <a href="#/">返回首页</a></p>}/>
</Routes></Suspense></DesktopShell></HashRouter></LocaleProvider></ThemeProvider></ErrorBoundary>);
