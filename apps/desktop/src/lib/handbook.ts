import { appFetch } from '@/transport/client';

export type HandbookKind = 'tutorial' | 'question-bank' | 'blog';

export interface HandbookCategory {
  id: string;
  title: string;
  titleEn: string;
  description: string;
}

export interface HandbookDocument {
  slug: string;
  title: string;
  titleEn: string;
  category: string;
  kind: HandbookKind;
  zhPath?: string;
  enPath?: string;
  htmlPath?: string;
  code?: string[];
  questionCount?: number;
  sourceFile?: string;
  sourceFileEn?: string;
}

export interface HandbookSource {
  id: string;
  name: string;
  author: string;
  url: string;
  commit: string;
  license: string;
}

export interface HandbookIndex {
  version: number;
  importedAt: string;
  title: string;
  subtitle: string;
  categories: HandbookCategory[];
  documents: HandbookDocument[];
  sources: HandbookSource[];
  stats: {
    tutorials: number;
    languages: number;
    codeExamples: number;
    blogs: number;
    embodiedVolumes: number;
    embodiedQuestions: number;
    interviewQuestions: number;
  };
}

export interface HandbookQuestion {
  id: string;
  documentSlug: string;
  documentTitle: string;
  category: string;
  kind: HandbookKind;
  difficulty: string;
  title: string;
  answer: string;
  href: string;
}

let indexPromise: Promise<HandbookIndex> | undefined;
let questionsPromise: Promise<{ questions: HandbookQuestion[] }> | undefined;

export function loadHandbookIndex() {
  indexPromise ??= appFetch('/handbook/index.json').then((response) => {
    if (!response.ok) throw new Error(`理论手册索引加载失败 (${response.status})`);
    return response.json() as Promise<HandbookIndex>;
  });
  return indexPromise;
}

export function loadHandbookQuestions() {
  questionsPromise ??= appFetch('/handbook/questions.json').then((response) => {
    if (!response.ok) throw new Error(`面试题索引加载失败 (${response.status})`);
    return response.json() as Promise<{ questions: HandbookQuestion[] }>;
  });
  return questionsPromise;
}

export function handbookDocumentPath(document: HandbookDocument, locale: 'zh' | 'en') {
  if (locale === 'en' && document.enPath) return document.enPath;
  return document.zhPath ?? document.enPath;
}

export function handbookHref(document: HandbookDocument) {
  return `/bagu/${document.kind}/${document.slug}`;
}

export function handbookAsset(path: string) {
  return `/handbook/${path.replace(/^\/+/, '')}`;
}
