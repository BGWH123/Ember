import { useLocation,useNavigate,useSearchParams as search,useParams as routerParams } from 'react-router-dom';
export function useParams<T extends Record<string,string>>():T { return routerParams() as T; }
export const usePathname=()=>useLocation().pathname;
export const useSearchParams=()=>search()[0];
export function useRouter(){const navigate=useNavigate();return {push:(url:string)=>navigate(url),replace:(url:string)=>navigate(url,{replace:true}),back:()=>navigate(-1)};}
