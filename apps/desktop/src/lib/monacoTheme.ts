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
    'editor.background': '#ffffff',
    'editor.foreground': '#20292f',
    'editor.lineHighlightBackground': '#f4f7f8',
    'editor.selectionBackground': '#0f766e22',
    'editorLineNumber.foreground': '#c7c7cc',
    'editorLineNumber.activeForeground': '#6e6e73',
    'editor.inactiveSelectionBackground': '#0f766e11',
    'editorIndentGuide.background': '#e5e5e5',
    'editorCursor.foreground': '#0f766e',
    'editorSuggestWidget.background': '#ffffff',
    'editorSuggestWidget.foreground': '#20292f',
    'editorSuggestWidget.selectedBackground': '#f4f7f8',
    'editorSuggestWidget.highlightForeground': '#0f766e',
    'editorSuggestWidget.border': '#e5e5e5',
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
    'editor.background': '#151b20',
    'editor.foreground': '#e5e5e5',
    'editor.lineHighlightBackground': '#1e2830',
    'editor.selectionBackground': '#5ee0cc22',
    'editorLineNumber.foreground': '#555555',
    'editorLineNumber.activeForeground': '#8e8e93',
    'editor.inactiveSelectionBackground': '#5ee0cc11',
    'editorIndentGuide.background': '#333333',
    'editorCursor.foreground': '#5ee0cc',
    'editorSuggestWidget.background': '#151b20',
    'editorSuggestWidget.foreground': '#e5e5e5',
    'editorSuggestWidget.selectedBackground': '#1e2830',
    'editorSuggestWidget.highlightForeground': '#5ee0cc',
    'editorSuggestWidget.border': '#333333',
  },
};
