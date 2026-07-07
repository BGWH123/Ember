import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['var(--font-sans)'],
        mono: ['var(--font-mono)'],
      },
      colors: {
        bg: {
          DEFAULT: 'var(--bg)',
          elev: 'var(--bg-elev)',
          sunken: 'var(--bg-sunken)',
          code: 'var(--bg-code)',
        },
        line: {
          DEFAULT: 'var(--line)',
          strong: 'var(--line-strong)',
        },
        text: {
          DEFAULT: 'var(--text)',
          '2': 'var(--text-2)',
          '3': 'var(--text-3)',
        },
        accent: {
          DEFAULT: 'var(--accent)',
          ink: 'var(--accent-ink)',
          wash: 'var(--accent-wash)',
          line: 'var(--accent-line)',
        },
        easy: 'var(--easy)',
        medium: 'var(--medium)',
        hard: 'var(--hard)',
      },
      boxShadow: {
        soft: '0 1px 3px rgba(0,0,0,0.06)',
        'soft-lg': '0 4px 12px rgba(0,0,0,0.08)',
      },
      borderColor: {
        DEFAULT: 'var(--line)',
      },
      borderRadius: {
        pill: '9999px',
      },
      animation: {
        ticker: 'ticker 60s linear infinite',
      },
      keyframes: {
        ticker: {
          from: { transform: 'translateX(0)' },
          to: { transform: 'translateX(-50%)' },
        },
      },
    },
  },
  plugins: [],
};
export default config;
