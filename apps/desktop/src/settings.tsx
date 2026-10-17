import {useEffect,useState} from 'react';
import {invoke} from '@tauri-apps/api/core';
import {EmberActivity,WaitingState} from '@/components/ui/WaitingState';
interface Status {moduleError?:string;mode:string;available:boolean;activeVersion:string|null;installed:string[];catalogVersion:string;protocolVersion:number;target:string;shellVersion:string}
export default function Settings(){
  const [status,setStatus]=useState<Status>();const [error,setError]=useState('');const [busy,setBusy]=useState(false);
  const [url,setUrl]=useState('');const [manifest,setManifest]=useState('');const [archive,setArchive]=useState('');
  const [legacy,setLegacy]=useState('');const [users,setUsers]=useState<string[]>([]);const [user,setUser]=useState('');const [notice,setNotice]=useState('');const [update,setUpdate]=useState<{version:string}|null>(null);
  const refresh=()=>invoke<Status>('evaluator_status').then(setStatus);
  useEffect(()=>{refresh().catch(e=>setError(String(e)));},[]);
  async function action(work:()=>Promise<unknown>){setBusy(true);setError('');setNotice('');try{await work();await refresh();}catch(e){setError(String(e));}finally{setBusy(false);}}
  const input='field rounded-lg px-3 py-2.5 bg-bg w-full text-sm';const button='inline-flex items-center gap-2 border border-line rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-40 hover:bg-bg-sunken transition-colors';
  return <><main className="max-w-3xl mx-auto px-7 py-10 space-y-6"><div className="mb-8"><p className="eyebrow mb-2">WORKSPACE</p><h1 className="text-2xl font-semibold tracking-tight">设置</h1><p className="text-sm text-text-2 mt-2">管理评测环境、学习记录与应用更新。</p></div>
    {error&&<p role="alert" className="text-red-500 whitespace-pre-wrap">{error}</p>}{notice&&<p role="status">{notice}</p>}
    <section className="card p-6 space-y-4 [&>p]:text-sm [&>p]:leading-7 [&>p]:text-text-2"><h2 className="text-xl">评测器</h2><p>未安装评测器时，题库、学习路线与参考答案仍可使用。首次评测需要准备运行环境。</p>
      {status?.moduleError&&<p role="alert">{status.moduleError}</p>}
      {status?<p className="flex items-center gap-2"><span className={`h-1.5 w-1.5 rounded-full ${status.available?'bg-easy':'bg-text-3'}`} />{`当前：${status.mode} · ${status.available?'可用':'未安装'} · ${status.activeVersion??'无本地版本'}`}</p>:<WaitingState compact label="读取评测器状态" description="正在检查本地运行环境。" />}
      <input aria-label="签名清单地址" className={input} placeholder="评测模块签名清单 URL" value={url} onChange={e=>setUrl(e.target.value)}/>
      <button className={button} disabled={busy||!url} onClick={()=>action(()=>invoke('install_evaluator',{url}))}>{busy&&<EmberActivity small />}{busy?'正在处理…':'下载并启用评测模块'}</button>
      <details><summary>离线安装</summary><div className="space-y-3 pt-3"><input className={input} placeholder="签名清单文件的绝对路径" value={manifest} onChange={e=>setManifest(e.target.value)}/><input className={input} placeholder="模块 ZIP 的绝对路径" value={archive} onChange={e=>setArchive(e.target.value)}/><button className={button} disabled={busy||!manifest||!archive} onClick={()=>action(()=>invoke('import_evaluator',{manifest,archive}))}>安装离线模块</button></div></details>
      {!!status?.installed.length&&<div className="flex gap-3 flex-wrap">{status.installed.map(version=><button className={button} key={version} disabled={busy||version===status.activeVersion} onClick={()=>action(()=>invoke('activate_evaluator',{version}))}>切换至 {version}</button>)}</div>}
    </section>
    <section className="card p-6 space-y-4 [&>p]:text-sm [&>p]:leading-7 [&>p]:text-text-2"><h2 className="text-xl">导入原版学习记录</h2><p>选择旧 ember.db 和用户。旧数据库保持只读，每个来源用户只能导入一次。</p><input className={input} placeholder="旧数据库绝对路径" value={legacy} onChange={e=>setLegacy(e.target.value)}/><button className={button} disabled={busy||!legacy} onClick={()=>action(async()=>{const list=await invoke<string[]>('legacy_users',{path:legacy});setUsers(list);setUser(list[0]??'');})}>读取用户</button>
    {!!users.length&&<><select className={input} value={user} onChange={e=>setUser(e.target.value)}>{users.map(u=><option key={u}>{u}</option>)}</select><button className={button} disabled={busy||!user} onClick={()=>action(async()=>{const result=await invoke('import_legacy',{path:legacy,user});setNotice(`导入完成：${JSON.stringify(result)}`);})}>导入所选用户</button></>}</section>
    <section className="card p-6 space-y-4 [&>p]:text-sm [&>p]:leading-7 [&>p]:text-text-2"><h2 className="text-xl">应用更新</h2><p>应用版本 {status?.shellVersion}。应用和评测模块分别更新；更新应用不会删除学习记录。</p><button className={button} disabled={busy} onClick={()=>action(async()=>{const next=await invoke<{version:string}|null>('check_shell_update');setUpdate(next);setNotice(next?`可更新至 ${next.version}`:'已是最新版本');})}>检查应用更新</button>{update&&<button className={button} disabled={busy} onClick={()=>action(async()=>{await invoke('install_shell_update');setNotice('更新已安装，请关闭后重新启动应用。');})}>安装 {update.version}</button>}</section>
  </main></>;
}
