import type * as Monaco from 'monaco-editor/esm/vs/editor/editor.api';

const KEYWORDS = [
  'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue',
  'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in',
  'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield',
];

const BUILTINS = ['abs', 'all', 'any', 'enumerate', 'isinstance', 'len', 'list', 'max', 'min', 'print', 'range', 'sum', 'zip'];

const TORCH = [
  'abs', 'add', 'arange', 'argmax', 'argmin', 'argsort', 'bmm', 'cat', 'chunk', 'clamp', 'concat',
  'cumsum', 'diag', 'einsum', 'empty', 'exp', 'eye', 'flatten', 'full', 'gather', 'gelu', 'linspace',
  'log', 'log_softmax', 'matmul', 'mean', 'mm', 'norm', 'ones', 'permute', 'pow', 'randn', 'relu',
  'reshape', 'sigmoid', 'silu', 'softmax', 'split', 'sqrt', 'squeeze', 'stack', 'std', 'sum', 'tanh',
  'tensor', 'topk', 'transpose', 'tril', 'triu', 'unsqueeze', 'var', 'where', 'zeros',
];

const NN = [
  'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AvgPool2d', 'BatchNorm1d', 'BatchNorm2d', 'Conv1d',
  'Conv2d', 'Dropout', 'Embedding', 'GELU', 'GRU', 'GroupNorm', 'Identity', 'LSTM', 'LayerNorm',
  'Linear', 'MaxPool2d', 'Module', 'ModuleDict', 'ModuleList', 'MultiheadAttention', 'Parameter',
  'ReLU', 'RNN', 'Sequential', 'SiLU', 'Softmax', 'Transformer',
];

const FUNCTIONAL = [
  'adaptive_avg_pool2d', 'avg_pool2d', 'batch_norm', 'binary_cross_entropy',
  'binary_cross_entropy_with_logits', 'conv1d', 'conv2d', 'cosine_similarity', 'cross_entropy',
  'dropout', 'embedding', 'gelu', 'interpolate', 'layer_norm', 'linear', 'log_softmax', 'max_pool2d',
  'mse_loss', 'pad', 'relu', 'scaled_dot_product_attention', 'sigmoid', 'silu', 'softmax', 'tanh',
];

const NUMPY = ['abs', 'arange', 'array', 'concatenate', 'eye', 'linspace', 'mean', 'ones', 'reshape', 'sqrt', 'stack', 'sum', 'zeros'];

const MEMBERS: Record<string, readonly string[]> = {
  torch: TORCH,
  'torch.nn': NN,
  nn: NN,
  'torch.nn.functional': FUNCTIONAL,
  F: FUNCTIONAL,
  np: NUMPY,
  numpy: NUMPY,
};

let registered = false;

export function registerPythonCompletions(monaco: typeof Monaco) {
  if (registered) return;
  registered = true;
  monaco.languages.setLanguageConfiguration('python', {
    comments: { lineComment: '#' },
    brackets: [['(', ')'], ['[', ']'], ['{', '}']],
    autoClosingPairs: [
      { open: '(', close: ')' },
      { open: '[', close: ']' },
      { open: '{', close: '}' },
      { open: '"', close: '"', notIn: ['string', 'comment'] },
      { open: "'", close: "'", notIn: ['string', 'comment'] },
    ],
    surroundingPairs: [
      { open: '(', close: ')' },
      { open: '[', close: ']' },
      { open: '{', close: '}' },
      { open: '"', close: '"' },
      { open: "'", close: "'" },
    ],
  });
  monaco.languages.registerCompletionItemProvider('python', {
    triggerCharacters: ['.'],
    provideCompletionItems(model, position) {
      const line = model.getLineContent(position.lineNumber).slice(0, position.column - 1);
      const member = /([A-Za-z_][\w.]*)\.[A-Za-z_]*$/.exec(line);
      const word = model.getWordUntilPosition(position);
      const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn,
      };
      const names = member ? MEMBERS[member[1]] ?? [] : [...KEYWORDS, ...BUILTINS, 'torch', 'nn', 'F', 'np'];
      const kindFor = (name: string) => {
        if (KEYWORDS.includes(name)) return monaco.languages.CompletionItemKind.Keyword;
        if (name === 'torch' || name === 'nn' || name === 'F' || name === 'np') return monaco.languages.CompletionItemKind.Module;
        return name[0] === name[0]?.toUpperCase()
          ? monaco.languages.CompletionItemKind.Class
          : monaco.languages.CompletionItemKind.Function;
      };
      return {
        suggestions: names.map((name) => ({
          label: name,
          kind: kindFor(name),
          insertText: name,
          range,
        })),
      };
    },
  });
}
