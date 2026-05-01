'use client';

import { useEffect } from 'react';

/**
 * Applies saved theme/design preferences from localStorage on mount.
 * Extracted as a client component to avoid SSR hydration mismatch in layout.tsx.
 */
export function ThemeScript() {
  useEffect(() => {
    try {
      const t = localStorage.getItem('pyre-theme');
      if (t === 'dark') document.documentElement.setAttribute('data-theme', 'dark');
      const d = localStorage.getItem('pyre-design');
      if (d === 'classic') document.documentElement.setAttribute('data-design', 'classic');
    } catch (e) {
      // localStorage not available (e.g. SSR edge case)
    }
  }, []);

  return null;
}
