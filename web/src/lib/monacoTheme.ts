// Monaco theme definitions. We type these against a minimal local structural
// type instead of importing from `monaco-editor` directly — that package is only
// a transitive peer of `@monaco-editor/react` and isn't installed as a top-level
// dependency, so importing its types would break `tsc`/`next build`. The shape
// below matches `editor.IStandaloneThemeData` closely enough for our usage and
// is accepted by `monaco.editor.defineTheme` at runtime.
interface MonacoThemeData {
  base: 'vs' | 'vs-dark' | 'hc-black' | 'hc-light';
  inherit: boolean;
  rules: Array<{ token: string; foreground?: string; background?: string; fontStyle?: string }>;
  colors: Record<string, string>;
}

export const appleLight: MonacoThemeData = {
  base: 'vs',
  inherit: true,
  rules: [
    { token: 'comment', foreground: 'aeaeb2', fontStyle: 'italic' },
    { token: 'keyword', foreground: 'af52de' },
    { token: 'string', foreground: '34c759' },
    { token: 'number', foreground: 'ff9500' },
    { token: 'type', foreground: '007aff' },
    { token: 'function', foreground: '007aff' },
    { token: 'variable', foreground: '1d1d1f' },
    { token: 'operator', foreground: '6e6e73' },
    { token: 'delimiter', foreground: '6e6e73' },
  ],
  colors: {
    'editor.background': '#fafafa',
    'editor.foreground': '#1d1d1f',
    'editor.lineHighlightBackground': '#f0f0f0',
    'editor.selectionBackground': '#007aff22',
    'editorLineNumber.foreground': '#c7c7cc',
    'editorLineNumber.activeForeground': '#6e6e73',
    'editor.inactiveSelectionBackground': '#007aff11',
    'editorIndentGuide.background': '#e5e5e5',
    'editorCursor.foreground': '#007aff',
  },
};

export const appleDark: MonacoThemeData = {
  base: 'vs-dark',
  inherit: true,
  rules: [
    { token: 'comment', foreground: '6e6e73', fontStyle: 'italic' },
    { token: 'keyword', foreground: 'c084fc' },
    { token: 'string', foreground: '6ee7b7' },
    { token: 'number', foreground: 'fbbf24' },
    { token: 'type', foreground: '4ea1ff' },
    { token: 'function', foreground: '4ea1ff' },
    { token: 'variable', foreground: 'e5e5e5' },
    { token: 'operator', foreground: '8e8e93' },
    { token: 'delimiter', foreground: '8e8e93' },
  ],
  colors: {
    'editor.background': '#1e1e1e',
    'editor.foreground': '#e5e5e5',
    'editor.lineHighlightBackground': '#2a2a2a',
    'editor.selectionBackground': '#4ea1ff33',
    'editorLineNumber.foreground': '#555555',
    'editorLineNumber.activeForeground': '#8e8e93',
    'editor.inactiveSelectionBackground': '#4ea1ff18',
    'editorIndentGuide.background': '#333333',
    'editorCursor.foreground': '#4ea1ff',
  },
};
