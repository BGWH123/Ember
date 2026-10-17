
import { useParams } from '@/routing/navigation';
import Link from '@/routing/Link';
import { TopNav } from '@/components/layout/TopNav';
import { Footer } from '@/components/layout/Footer';
import { SolutionPageContent } from '@/components/solution/SolutionPage';

export default function SolutionsPage() {
  const { id } = useParams<{ id: string }>();

  return (
    <div className="min-h-screen bg-bg">
      <TopNav />
      <main className="max-w-[1280px] mx-auto px-7 pt-8 pb-20">
        <SolutionPageContent problemId={id} />
      </main>
      <Footer />
    </div>
  );
}
