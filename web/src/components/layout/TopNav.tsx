'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Sun, Moon, SwatchBook, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useLocale } from '@/context/LocaleContext';
import { useTheme } from '@/context/ThemeContext';
import { useDesign } from '@/context/DesignContext';

interface TopNavProps {
  solvedCount?: number;
  totalCount?: number;
}

function EmberGlyph() {
  return (
    <span
      className="w-[22px] h-[22px] inline-flex items-center justify-center rounded-[6px] text-accent"
      style={{
        border: '1px solid var(--accent-line)',
        background: 'var(--accent-wash)',
      }}
    >
      <svg width="12" height="12" viewBox="0 0 12 12" fill="none" className="animate-pulse-glow">
        <path
          d="M6 1.25c.6 1.8.15 2.7-.75 3.6C4 6.2 3.25 7.2 3.25 8.5a2.75 2.75 0 1 0 5.5 0c0-1-.3-1.8-1-2.6.3 1.1-.15 1.8-.8 1.8-.5 0-.85-.4-.85-1C6.1 5.6 6.7 3.9 6 1.25Z"
          fill="currentColor"
        />
      </svg>
    </span>
  );
}

export function TopNav({ solvedCount, totalCount }: TopNavProps) {
  const pathname = usePathname();
  const { locale, setLocale, t } = useLocale();
  const { theme, toggleTheme } = useTheme();
  const { toggleDesign } = useDesign();

  const links = [
    { href: '/', label: t('home'), key: 'home' },
    { href: '/problems', label: t('problems'), key: 'problems' },
    { href: '/paths', label: t('paths'), key: 'paths' },
    { href: '/bagu', label: '八股文', key: 'bagu' },
    { href: '/interview', label: '模拟面试', key: 'interview' },
  ];

  const isActive = (href: string) => {
    if (href === '/') return pathname === '/';
    return pathname.startsWith(href);
  };

  return (
    <nav
      className="sticky top-0 z-50"
      style={{
        backdropFilter: 'saturate(180%) blur(20px)',
        WebkitBackdropFilter: 'saturate(180%) blur(20px)',
        background: 'color-mix(in oklab, var(--bg) 78%, transparent)',
        borderBottom: '1px solid var(--line)',
        boxShadow: '0 1px 0 0 var(--accent-glow), inset 0 -1px 0 0 var(--accent-glow)',
      }}
    >
      <div className="max-w-[1280px] mx-auto px-7 h-[52px] flex items-center justify-between gap-6">
        {/* Left: Logo + Links */}
        <div className="flex items-center gap-7">
          <Link
            href="/"
            className="inline-flex items-center gap-2.5 font-semibold text-[15px] tracking-[-0.01em] group"
          >
            <span className="transition-[filter] duration-300 group-hover:drop-shadow-[0_0_6px_var(--accent)]">
              <EmberGlyph />
            </span>
            <span className="relative">
              Ember
              <span
                className="absolute -bottom-[1px] left-0 h-[1.5px] bg-accent rounded-full origin-left transition-transform duration-300 scale-x-0 group-hover:scale-x-100"
                style={{ width: '100%' }}
              />
            </span>
          </Link>

          <div className="flex items-center gap-0.5">
            {links.map((link) => (
              <Link
                key={link.key}
                href={link.href}
                className={cn(
                  'relative px-2.5 py-1.5 rounded-lg text-[13px] transition-[color,background] duration-200',
                  isActive(link.href)
                    ? 'text-text'
                    : 'text-text-2 hover:text-text'
                )}
              >
                {link.label}
                {/* Active indicator dot */}
                {isActive(link.href) && (
                  <span
                    className="absolute bottom-[3px] left-1/2 -translate-x-1/2 w-1 h-1 rounded-full bg-accent"
                    style={{ boxShadow: '0 0 6px var(--accent-glow)' }}
                  />
                )}
                {/* Hover underline */}
                <span
                  className={cn(
                    'absolute -bottom-[1px] left-2.5 right-2.5 h-[1.5px] bg-accent rounded-full origin-left transition-transform duration-300',
                    isActive(link.href) ? 'scale-x-0' : 'scale-x-0 hover:scale-x-100'
                  )}
                />
              </Link>
            ))}
          </div>
        </div>

        {/* Right: Actions */}
        <div className="flex items-center gap-2">
          {solvedCount !== undefined && totalCount !== undefined && (
            <div
              className="hidden sm:inline-flex items-center gap-2 px-2.5 py-[5px] rounded-full mono text-[11px] text-text-2"
              style={{
                border: '1px solid var(--line)',
                background: 'var(--bg-elev)',
              }}
            >
              <Sparkles className="w-3 h-3 text-accent" />
              <span>{t('solvedCount', { solved: solvedCount, total: totalCount })}</span>
            </div>
          )}

          <button
            onClick={toggleDesign}
            className="w-8 h-8 inline-flex items-center justify-center rounded-lg text-text-2 cursor-pointer transition-all duration-200 hover:text-text hover:scale-105 active:scale-95"
            style={{
              border: '1px solid var(--line)',
              background: 'var(--bg-elev)',
            }}
            title="Switch design"
          >
            <SwatchBook className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={toggleTheme}
            className="w-8 h-8 inline-flex items-center justify-center rounded-lg text-text-2 cursor-pointer transition-all duration-200 hover:text-text hover:scale-105 active:scale-95"
            style={{
              border: '1px solid var(--line)',
              background: 'var(--bg-elev)',
            }}
            title="Toggle theme"
          >
            {theme === 'light' ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
          </button>

          <button
            onClick={() => setLocale(locale === 'en' ? 'zh' : 'en')}
            className="h-[30px] px-3 inline-flex items-center gap-1.5 rounded-lg text-[12.5px] text-text-2 cursor-pointer transition-all duration-200 hover:text-text hover:scale-105 active:scale-95"
            style={{
              border: '1px solid var(--line)',
              background: 'var(--bg-elev)',
            }}
          >
            {locale === 'en' ? 'EN' : '中文'}
          </button>
        </div>
      </div>
    </nav>
  );
}
