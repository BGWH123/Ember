
import { useParams } from '@/routing/navigation';
import Link from '@/routing/Link';
import { SolutionPageContent } from '@/components/solution/SolutionPage';

export default function SolutionsPage() {
  const { id } = useParams<{ id: string }>();

  return (
    <div className="desktop-page bg-bg">

      <main className="max-w-[1280px] mx-auto px-7 pt-6 pb-8">
        <SolutionPageContent problemId={id} />
      </main>

    </div>
  );
}
