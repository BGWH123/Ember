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
  child=spawn(join(root,'target/release/ember-headless'+(process.platform==='win32'?'.exe':'')),[],{cwd:root,env:{...process.env,EMBER_ROOT:root,EMBER_DATA:directory,EMBER_DEV_PYTHON:process.env.EMBER_DEV_PYTHON??join(root,process.platform==='win32'?'.venv/Scripts/python.exe':'.venv/bin/python')}});
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
      const result=await command(name==='request'?{path:args.path,payload:args.payload}:{command:name==='evaluator_status'?'status':name,...args});
      if(!result.ok)throw new Error(result.error);return result.body;
    });chain=operation.then(()=>{},()=>{});return operation;
  });
  await page.addInitScript(()=>{
    localStorage.setItem('locale','en');
    (window as any).__TAURI_INTERNALS__={invoke:(name:string,args:unknown)=>(window as any).__EMBER_TEST_INVOKE__(name,args)};
  });
  // Font files must work from the packaged application origin, including Chinese.
  const fontRequests:string[]=[];
  page.on('request',request=>{if(request.resourceType()==='font')fontRequests.push(request.url());});
  await page.goto('/');await expect(page.getByRole('heading',{name:'Workspace',exact:true})).toBeVisible();
  await expect(page.getByRole('heading',{name:'Implement ReLU',exact:true})).toBeVisible();
  await expect(page.getByRole('banner')).toHaveCount(1);
  await expect(page.getByRole('button',{name:'Minimize',exact:true})).toBeDisabled(); // Native window controls are verified in the installed application.
  await page.locator('.desktop-shell').evaluate(element=>element.setAttribute('data-persistent','true'));
  const loadedFonts=await page.evaluate(async()=>Promise.all([
    document.fonts.load('16px "Noto Sans Variable"','Ember'),
    document.fonts.load('16px "Noto Sans SC Variable"','中文题库'),
    document.fonts.load('14px "Fira Code Variable"','def relu(x): return x >= 0'),
  ]).then(groups=>groups.map(faces=>faces.length)));
  expect(loadedFonts.every(count=>count>0)).toBe(true);
  await page.screenshot({path:join(root,'artifacts/desktop/home-refreshed.png'),fullPage:true});
  for(const label of ['Problems','Learning paths','Knowledge map','Theory handbook','Hot 50','Mock interview']){
    await page.getByRole('complementary',{name:'App navigation'}).getByRole('link',{name:label,exact:true}).click();
    await expect(page.locator('main').first()).toBeVisible();
    await expect(page.locator('.desktop-shell')).toHaveAttribute('data-persistent','true');
    await expect(page.getByRole('banner')).toHaveCount(1);
  }
  await page.goto('/#/bagu');
  await expect(page.getByRole('heading',{name:'Ember AI Theory Handbook',exact:true})).toBeVisible();
  await expect(page.getByText('35',{exact:true}).first()).toBeVisible();
  await page.getByRole('link',{name:/Attention/}).first().click();
  await expect(page.getByRole('heading',{name:'Attention',exact:true})).toBeVisible();
  await expect(page.locator('.handbook-markdown').first()).toContainText('Attention');
  await page.goto('/#/bagu/question-bank/embodied-basics');
  await expect(page.getByRole('heading',{name:'Embodied Basics',exact:true})).toBeVisible();
  await expect(page.locator('.handbook-markdown details').first()).toBeVisible();
  await page.goto('/#/bagu/blog/cosmos3-mot');
  await expect(page.locator('iframe[title*=\"Cosmos 3\"]')).toBeVisible();
  await page.goto('/#/problems/relu');
  await expect(page.getByRole('tab',{name:'relu.py',exact:true})).toBeVisible();
  await page.goto('/#/problems/sigmoid');
  await expect(page.getByRole('tab',{name:'relu.py',exact:true})).toBeVisible();
  await expect(page.getByRole('tab',{name:'sigmoid.py',exact:true})).toBeVisible();
  await page.getByRole('tab',{name:'relu.py',exact:true}).click();
  await expect(page).toHaveURL(/#\/problems\/relu$/);
  const editor=page.locator('.monaco-editor textarea').first();await expect(editor).toBeVisible();
  await editor.focus();await page.keyboard.press('Control+a');await page.keyboard.insertText('def relu(x):\n    return x.clamp(min=0)\n');
  await page.getByRole('button',{name:/^Run/}).click();
  await expect(page.getByRole('status')).toContainText('Running');
  await expect(page.getByRole('status')).toHaveCSS('opacity','1');
  await page.screenshot({path:join(root,'artifacts/desktop/evaluation-waiting.png'),fullPage:true});
  await expect(page.getByRole('button',{name:/^Run/})).toBeEnabled({timeout:30000});
  await page.getByRole('button',{name:'Submit',exact:true}).click();
  await expect(page.getByRole('button',{name:'Submit',exact:true})).toBeEnabled({timeout:30000});
  const history=await command({path:'/api/submissions/relu'});expect(history.body).toHaveLength(2);
  const progress=await command({path:'/api/progress'});expect(progress.body.progress.relu?.status,JSON.stringify(history.body)).toBe('solved');
  await page.screenshot({path:join(root,'artifacts/desktop/workspace.png'),fullPage:true});
  // Dismiss the completion dialog before navigating.
  await page.keyboard.press('Escape');
  await page.goto('/#/settings');await expect(page.getByRole('heading',{name:'评测器',exact:true})).toBeVisible();
  await expect(page.getByRole('heading',{name:'在线评测',exact:true})).toHaveCount(0);
  await expect(page.getByRole('button',{name:'启用在线评测',exact:true})).toHaveCount(0);
  await page.goto('/#/problems/relu');await expect(page.locator('.monaco-editor').first()).toContainText('clamp');
  await page.getByRole('button',{name:'Toggle sidebar',exact:true}).click();
  await expect(page.locator('.desktop-shell')).toHaveClass(/desktop-shell--collapsed/);
  await page.getByRole('button',{name:'Toggle sidebar',exact:true}).click();
  await expect(page.locator('.desktop-shell')).not.toHaveClass(/desktop-shell--collapsed/);
  await page.locator('button').filter({has:page.locator('svg.lucide-sun')}).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme','dark');
  await expect(page.locator('.monaco-editor').first()).toHaveClass(/vs-dark/);
  await page.screenshot({path:join(root,'artifacts/desktop/workspace-dark.png'),fullPage:true});
  await page.emulateMedia({reducedMotion:'reduce'});
  await page.getByRole('button',{name:/^Run/}).click();
  await expect(page.getByRole('status')).toContainText('Running');
  await expect(page.locator('.waiting-state .ember-activity__matrix > span').first()).toHaveCSS('animation-name','none');
  await expect(page.getByRole('button',{name:/^Run/})).toBeEnabled({timeout:30000});
  await page.setViewportSize({width:900,height:600});
  await expect(page.locator('.desktop-shell')).toHaveClass(/desktop-shell--collapsed/);
  await expect(page.getByRole('button',{name:'Submit',exact:true})).toBeInViewport();
  const viewport=await page.evaluate(()=>({height:innerHeight,documentHeight:document.documentElement.scrollHeight,width:innerWidth,documentWidth:document.documentElement.scrollWidth}));
  expect(viewport.documentHeight).toBe(viewport.height);expect(viewport.documentWidth).toBe(viewport.width);
  await page.screenshot({path:join(root,'artifacts/desktop/workspace-compact.png'),fullPage:true});
  expect(fontRequests.length).toBeGreaterThan(0);
  expect(fontRequests.every(url=>url.startsWith('http://127.0.0.1:1420/'))).toBe(true);
  expect(errors).toEqual([]);
});
