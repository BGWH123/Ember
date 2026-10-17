
import { HomeContent } from '@/components/home/HomeContent';
import {useEffect,useState} from 'react';
import {appFetch} from '@/transport/client';

export default function HomePage() {
  const [stats,setStats]=useState({total:0,easy:0,medium:0,hard:0});
  useEffect(()=>{appFetch('/api/stats').then(r=>r.json()).then(setStats);},[]);

  return (
    <div className="desktop-page bg-bg">

      <HomeContent stats={stats} />

    </div>
  );
}
