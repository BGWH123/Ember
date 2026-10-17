import { invoke } from '@tauri-apps/api/core';
// A presentation adapter for migrated pages, not an HTTP server or global fetch patch.
export async function appFetch(input:string,init?:RequestInit):Promise<Response>{
  if(!input.startsWith('/api/')) return fetch(input,init);
  try {
    const payload=typeof init?.body==='string'?JSON.parse(init.body):null;
    const body=await invoke('request',{path:input,payload});
    return Response.json(body);
  }catch(error){return Response.json({error:String(error),message:String(error)},{status:400});}
}
