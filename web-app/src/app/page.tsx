import HeroSection from '@/components/HeroSection';
import ConceptsSection from '@/components/ConceptsSection';
import DashboardSection from '@/components/DashboardSection';
import SideNav from '@/components/SideNav';

export default function Home() {
  return (
    <>
      <SideNav />
      <main>
        {/* Zone A: Editorial Intro */}
        <HeroSection />
        <ConceptsSection />
        {/* Zone B: Interactive Dashboard */}
        <DashboardSection />
      </main>
    </>
  );
}
