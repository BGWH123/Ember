import {test,expect} from '@playwright/test';
import {spawn,type ChildProcessWithoutNullStreams} from 'node:child_process';
import {createInterface} from 'node:readline';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {resolve,join} from 'node:path';
let child:ChildProcessWithoutNullStreams;let directory:string;let pending:Array<(value:unknown)=>void>=[];
let chain=Promise.resolve();
const root=resolve('../..');
function command(value:unknown):Promise<any>{
  return new Promise((resolve,reject)=>{pending.push(resolve);child.stdin.write(JSON.stringify(value)+'\n',error=>{if(error)reject(error);});});
}
test.beforeAll(async()=>{
  directory=await mkdtemp(join(tmpdir(),'ember-ui-'));
  child=spawn(join(root,'target/release/ember-headless'),[],{cwd:root,env:{...process.env,EMBER_ROOT:root,EMBER_DATA:directory,EMBER_DEV_PYTHON:join(root,'.venv/bin/python')}});
  child.stderr.on('data',data=>process.stderr.write(data));
  const ready=new Promise<void>(resolve=>pending.push(()=>resolve()));
  createInterface({input:child.stdout}).on('line',line=>{pending.shift()?.(JSON.parse(line));});
  await ready;
});
test.afterAll(async()=>{if(child){await command({command:'shutdown'});child.stdin.end();await new Promise<void>(resolve=>child.on('exit',()=>resolve()));}if(directory)await rm(directory,{recursive:true,force:true});});
test('offline pages, real Rust/PyTorch submission, draft and navigation',async({page})=>{
  const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));
  await page.exposeFunction('__EMBER_TEST_INVOKE__',(name:string,args:Record<string,unknown>)=>{
    // Replaces only the native transport; all domain/storage/evaluation work is real.
    const operation=chain.then(async()=>{
      const result=await command(name==='request'?{path:args.path,payload:args.payload}:{command:name,...args});
      if(!result.ok)throw new Error(result.error);return result.body;
    });chain=operation.then(()=>{},()=>{});return operation;
  });
  await page.addInitScript(()=>{
    localStorage.setItem('locale','en');
    (window as any).__TAURI_INTERNALS__={invoke:(name:string,args:unknown)=>(window as any).__EMBER_TEST_INVOKE__(name,args)};
  });
  await page.goto('/');await expect(page.getByRole('heading',{name:'Implement the internals of modern AI systems'})).toBeVisible();
  for(const path of ['problems','paths','curriculum','bagu','hot','interview']){
    await page.goto('/#/'+path);await page.waitForTimeout(150);await expect(page.locator('main').first()).toBeVisible();
  }
  await page.goto('/#/problems/relu');
  const editor=page.locator('.monaco-editor textarea').first();await expect(editor).toBeVisible();
  await editor.focus();await page.keyboard.press('Control+a');await page.keyboard.insertText('def relu(x):\n    return x.clamp(min=0)\n');
  await page.getByRole('button',{name:/^Run/}).click();
  await expect(page.getByRole('button',{name:/^Run/})).toBeEnabled({timeout:30000});
  await page.getByRole('button',{name:'Submit',exact:true}).click();
  await expect(page.getByRole('button',{name:'Submit',exact:true})).toBeEnabled({timeout:30000});
  const progress=await command({path:'/api/progress'});expect(progress.body.progress.relu.status).toBe('solved');
  const history=await command({path:'/api/submissions/relu'});expect(history.body).toHaveLength(2);
  await page.screenshot({path:join(root,'artifacts/desktop/workspace.png'),fullPage:true});
  // Dismiss the completion dialog before navigating.
  await page.keyboard.press('Escape');
  await page.goto('/#/settings');await expect(page.getByRole('heading',{name:'评测器',exact:true})).toBeVisible();
  await page.goto('/#/problems/relu');await expect(page.locator('.monaco-editor').first()).toContainText('clamp');
  expect(errors).toEqual([]);
});
