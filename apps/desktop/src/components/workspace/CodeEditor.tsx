import * as monaco from 'monaco-editor/esm/vs/editor/editor.api';
import EditorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';
import 'monaco-editor/esm/vs/basic-languages/python/python.contribution';

import { useRef, useEffect } from 'react';
import Editor, { loader, type OnMount } from '@monaco-editor/react';
import { appleLight, appleDark } from '@/lib/monacoTheme';
import { useTheme } from '@/context/ThemeContext';
import { WaitingState } from '@/components/ui/WaitingState';

self.MonacoEnvironment = { getWorker: () => new EditorWorker() };
loader.config({ monaco });

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  readOnly?: boolean;
  height?: string;
  allowParentScrollOnWheel?: boolean;
}

export function CodeEditor({
  value,
  onChange,
  readOnly = false,
  height = '100%',
  allowParentScrollOnWheel = false,
}: CodeEditorProps) {
  const editorRef = useRef<any>(null);
  const monacoRef = useRef<any>(null);
  const { theme } = useTheme();

  const handleMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;
    monaco.editor.defineTheme('apple-light', appleLight);
    monaco.editor.defineTheme('apple-dark', appleDark);
    monaco.editor.setTheme(theme === 'dark' ? 'apple-dark' : 'apple-light');
    // Refresh cached glyph widths once the bundled font finishes loading.
    void document.fonts.load('14px "Fira Code Variable"').then(() => monaco.editor.remeasureFonts());
  };

  useEffect(() => {
    if (monacoRef.current) {
      monacoRef.current.editor.setTheme(theme === 'dark' ? 'apple-dark' : 'apple-light');
    }
  }, [theme]);

  return (
    <Editor
      height={height}
      loading={<WaitingState compact label="Ember" />}
      language="python"
      value={value}
      onChange={(v) => onChange(v || '')}
      onMount={handleMount}
      options={{
        readOnly,
        fontSize: 14,
        lineHeight: 22,
        fontFamily: "'Fira Code Variable', 'Noto Sans SC Variable', monospace",
        fontLigatures: true,
        minimap: { enabled: false },
        scrollBeyondLastLine: false,
        padding: { top: 16, bottom: 16 },
        renderLineHighlight: 'line',
        smoothScrolling: true,
        cursorBlinking: 'smooth',
        cursorSmoothCaretAnimation: 'on',
        bracketPairColorization: { enabled: true },
        overviewRulerBorder: false,
        hideCursorInOverviewRuler: true,
        scrollbar: {
          alwaysConsumeMouseWheel: !allowParentScrollOnWheel,
          verticalScrollbarSize: 6,
          horizontalScrollbarSize: 6,
        },
      }}
    />
  );
}
