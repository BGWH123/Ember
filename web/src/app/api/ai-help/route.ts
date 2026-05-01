import { NextResponse } from 'next/server';
import type { AiHelpRequest } from '@/lib/types';

/* ── env defaults ── */
const ENV_BASE_URL = process.env.AI_HELP_BASE_URL?.trim() || '';
const ENV_API_KEY = process.env.AI_HELP_API_KEY?.trim() || '';
const ENV_MODEL = process.env.AI_HELP_MODEL?.trim() || '';

/* ── simple in-memory rate limiter ── */
const RATE_WINDOW_MS = 60_000;
const RATE_MAX = 10;
const hits = new Map<string, number[]>();

function isRateLimited(ip: string): boolean {
  const now = Date.now();
  const timestamps = (hits.get(ip) ?? []).filter((t) => now - t < RATE_WINDOW_MS);
  if (timestamps.length >= RATE_MAX) {
    hits.set(ip, timestamps);
    return true;
  }
  timestamps.push(now);
  hits.set(ip, timestamps);
  return false;
}

/* ── helpers ── */
function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, '');
}

function extractMessageContent(content: unknown): string {
  if (typeof content === 'string') return content.trim();
  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (typeof item === 'string') return item;
        if (item && typeof item === 'object' && 'type' in item && 'text' in item) {
          const part = item as { type?: string; text?: string };
          return part.type === 'text' && part.text ? part.text : '';
        }
        return '';
      })
      .join('\n')
      .trim();
  }
  return '';
}

function buildPrompt(request: AiHelpRequest) {
  const testsText = request.sampleTests
    .map((test, index) => `Sample Test ${index + 1}: ${test.name}\n${test.code}`)
    .join('\n\n');
  const hasUserCode = Boolean(request.userCode?.trim());
  const hasCustomPrompt = Boolean(request.customPrompt?.trim());

  const responseInstruction = request.locale === 'zh'
    ? hasCustomPrompt
      ? [
          '请用中文并使用 Markdown 输出。',
          '严格遵循用户提示词的要求来回答。',
          ...(hasUserCode ? ['如果给了学习者当前代码，请优先分析代码并给出修改建议。'] : []),
          '不要给完整可提交实现；如果需要举例，只给局部代码片段。',
        ].join('\n')
      : hasUserCode
      ? [
          '请用中文并使用 Markdown 输出。',
          '请优先分析学习者当前代码，而不是泛泛讲题。',
          '按以下结构回答：',
          '## 现有实现中做对了什么',
          '## 主要问题',
          '## 修改建议',
          '## 下一步验证',
          '如果需要举例，只给局部代码片段，不要给完整可提交实现。',
        ].join('\n')
      : [
          '请用中文并使用 Markdown 输出。',
          '按以下结构回答：',
          '## 核心思路',
          '## 实现步骤',
          '## 容易出错的地方',
          '## 建议先验证的测试',
          '不要给完整可提交实现；如果需要举例，只给局部代码片段。',
        ].join('\n')
    : hasCustomPrompt
      ? [
          'Reply in English and format the answer with Markdown.',
          'Follow the custom user prompt closely.',
          ...(hasUserCode ? ['If learner code is provided, analyze it first and give concrete modification suggestions.'] : []),
          'Do not provide a full submit-ready implementation. If you include examples, keep them to small local snippets.',
        ].join('\n')
    : hasUserCode
      ? [
          'Reply in English and format the answer with Markdown.',
          'Prioritize analyzing the learner code instead of giving a generic explanation.',
          'Use this structure:',
          '## What is already correct',
          '## Main issues',
          '## How to modify the code',
          '## What to verify next',
          'If you include examples, keep them to small local snippets, not a full submit-ready implementation.',
        ].join('\n')
      : [
          'Reply in English and format the answer with Markdown.',
          'Use this structure:',
          '## Core idea',
          '## Implementation steps',
          '## Common mistakes',
          '## Tests to validate first',
          'Do not provide a full submit-ready implementation. If you include examples, keep them to small local snippets.',
        ].join('\n');

  return [
    `Problem ID: ${request.problemId}`,
    `Problem Title: ${request.problemTitle}`,
    `Target Function: ${request.functionName}`,
    '',
    'Problem Description:',
    request.description,
    '',
    'Sample Tests:',
    testsText || 'No sample tests available.',
    ...(hasCustomPrompt ? ['', 'Custom User Prompt:', request.customPrompt!.trim()] : []),
    ...(!hasCustomPrompt ? ['', 'Reference Solution Code:', request.solutionCode || 'No reference solution available.'] : []),
    ...(request.userCode ? ['', 'Learner Current Code:', request.userCode] : []),
    '',
    responseInstruction,
  ].join('\n');
}

export async function POST(request: Request) {
  const ip = request.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || 'unknown';
  if (isRateLimited(ip)) {
    return NextResponse.json(
      { error: 'Too many requests. Please wait a moment before trying again.' },
      { status: 429 },
    );
  }

  const body = (await request.json()) as AiHelpRequest;

  // Server env takes priority; fall back to client config
  const baseUrl = ENV_BASE_URL || body.config?.baseUrl?.trim() || '';
  const apiKey = ENV_API_KEY || body.config?.apiKey?.trim() || '';
  const model = ENV_MODEL || body.config?.model?.trim() || '';

  if (!baseUrl || !apiKey || !model) {
    return NextResponse.json({ error: 'Missing baseUrl, apiKey, or model.' }, { status: 400 });
  }

  const payload = {
    model,
    stream: false,
    temperature: 0.4,
    messages: [
      {
        role: 'system',
        content: [
          'You are an AI tutor for PyTorch and deep learning coding exercises.',
          'Focus on hints, debugging guidance, conceptual explanations, and next steps.',
          'Do not provide a complete final answer or a full submit-ready implementation.',
          'When learner code is present, analyze the code first and give concrete modification suggestions.',
          'Always return Markdown with clear headings and short bullet lists where helpful.',
        ].join(' '),
      },
      {
        role: 'user',
        content: buildPrompt(body),
      },
    ],
  };

  try {
    const upstream = await fetch(`${normalizeBaseUrl(baseUrl)}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(payload),
    });

    const data = await upstream.json().catch(() => null);
    if (!upstream.ok) {
      if (upstream.status === 401) {
        return NextResponse.json({ error: 'Invalid API key. Please check your configuration.' }, { status: 401 });
      }
      if (upstream.status === 429) {
        return NextResponse.json({ error: 'API rate limit exceeded. Please try again later.' }, { status: 429 });
      }
      const errorMessage =
        data && typeof data === 'object' && 'error' in data && data.error && typeof data.error === 'object' && 'message' in data.error
          ? String(data.error.message)
          : `Upstream request failed with status ${upstream.status}.`;
      return NextResponse.json({ error: errorMessage }, { status: 502 });
    }

    const guidance = extractMessageContent(data?.choices?.[0]?.message?.content);
    if (!guidance) {
      return NextResponse.json({ error: 'Model returned an empty response.' }, { status: 502 });
    }

    return NextResponse.json({ guidance, model: data?.model || model });
  } catch {
    return NextResponse.json({ error: 'Failed to reach the configured AI endpoint.' }, { status: 502 });
  }
}
