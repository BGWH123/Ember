import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { basename, join, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, index, all) => {
  if (value.startsWith('--')) pairs.push([value.slice(2), all[index + 1]]);
  return pairs;
}, []));
const arisRoot = resolve(args.aris ?? '../.handbook-imports/ARIS-in-AI-Offer');
const embodiedRoot = resolve(args.embodied ?? '../.handbook-imports/embodied-interview-qa');
const outputRoot = resolve(args.output ?? 'apps/desktop/public/handbook');
const mathjaxRoot = resolve(args.mathjax ?? 'apps/desktop/node_modules/mathjax/es5');

const categories = [
  { id: 'general', title: '基础知识', titleEn: 'General', description: 'Attention、Transformer、评测、归一化、优化器、分词与 KL 散度。' },
  { id: 'post-training', title: '后训练与推理', titleEn: 'Post-Training & Reasoning', description: 'RLHF、DPO、GRPO、推理模型、在线蒸馏与参数高效微调。' },
  { id: 'llm-systems', title: '大模型架构与系统', titleEn: 'LLM Architecture & Systems', description: 'MoE、长上下文、高效注意力、KV Cache、量化、分布式训练与推理服务。' },
  { id: 'generative-theory', title: '生成模型理论', titleEn: 'Generative Models', description: 'Flow Matching、Diffusion 基础与离散潜变量模型。' },
  { id: 'generation-systems', title: '生成系统', titleEn: 'Generation Systems', description: '图像、视频、3D 生成，以及 Diffusion 后训练与蒸馏。' },
  { id: 'multimodal', title: '多模态', titleEn: 'Multimodal', description: '视觉语言模型与多模态表示学习。' },
  { id: 'agents', title: '智能体', titleEn: 'Agents', description: 'Agent 基础、Agentic RL、多智能体、自进化与 RAG。' },
  { id: 'embodied-world-models', title: '具身智能与世界模型', titleEn: 'Embodied AI & World Models', description: '世界模型，以及完整的具身智能社区面试题库。' },
  { id: 'blogs', title: '技术长文', titleEn: 'Technical Blogs', description: '连续扩散语言模型、表征与流形、Cosmos 3 深度导读。' },
];

const tutorials = [
  ['attention', 'Attention 面试手册', 'Attention', 'general', 'attention_tutorial'],
  ['transformer-block', 'Transformer Block', 'Transformer Block', 'general', 'transformer_block_tutorial'],
  ['llm-evaluation', '大模型评测与基准', 'LLM Evaluation & Benchmarking', 'general', 'llm_evaluation_benchmarking_tutorial'],
  ['normalization-init', '归一化、残差与初始化', 'Normalization, Residuals & Initialization', 'general', 'normalization_init_tutorial'],
  ['optimizer-lr', '优化器与学习率调度', 'Optimizers & LR Schedules', 'general', 'optimizer_lr_schedule_tutorial'],
  ['tokenization', 'Tokenization 分词器', 'Tokenization', 'general', 'tokenization_tutorial'],
  ['kl-divergence-rlhf', 'RLHF 中的 KL 散度', 'KL Divergence in RLHF', 'general', 'kl_divergence_rlhf_tutorial'],
  ['rlhf-dpo-grpo-ppo', 'RLHF、DPO、GRPO 与 PPO', 'RLHF, DPO, GRPO & PPO', 'post-training', 'rlhf_dpo_grpo_ppo_tutorial'],
  ['reasoning-models', '推理模型', 'Reasoning Models', 'post-training', 'reasoning_models_tutorial'],
  ['llm-on-policy-distillation', '大模型在线策略蒸馏', 'LLM On-Policy Distillation', 'post-training', 'llm_opd_tutorial'],
  ['lora-peft', 'LoRA 与参数高效微调', 'LoRA & PEFT', 'post-training', 'lora_peft_tutorial'],
  ['moe', '混合专家模型 MoE', 'Mixture of Experts', 'llm-systems', 'moe_tutorial'],
  ['long-context', '长上下文', 'Long Context', 'llm-systems', 'long_context_rope_yarn_mla_tutorial'],
  ['linear-sparse-attention', '线性与稀疏注意力', 'Linear & Sparse Attention', 'llm-systems', 'linear_sparse_attention_tutorial'],
  ['kv-cache-speculative', 'KV Cache 与推测解码', 'KV Cache & Speculative Decoding', 'llm-systems', 'kv_cache_speculative_decoding_tutorial'],
  ['quantization', '模型量化', 'Quantization', 'llm-systems', 'quantization_tutorial'],
  ['distributed-training', '分布式训练', 'Distributed Training', 'llm-systems', 'distributed_training_tutorial'],
  ['llm-pretraining', '大模型预训练流水线', 'LLM Pretraining Pipeline', 'llm-systems', 'llm_pretraining_pipeline_tutorial'],
  ['llm-inference-serving', '大模型推理与服务栈', 'LLM Inference & Serving Stack', 'llm-systems', 'llm_inference_serving_tutorial'],
  ['flow-matching', 'Flow Matching 速查', 'Flow Matching Quick Reference', 'generative-theory', 'flow_matching_tutorial'],
  ['diffusion-foundations', 'Diffusion 基础', 'Diffusion Foundations', 'generative-theory', 'diffusion_foundations_tutorial'],
  ['vae-vqvae-vqgan', 'VAE、VQ-VAE、VQ-GAN 与 FSQ', 'VAE, VQ-VAE, VQ-GAN & FSQ', 'generative-theory', 'vae_vqvae_vqgan_tutorial'],
  ['image-generation', '图像生成系统', 'Image Generation Systems', 'generation-systems', 'image_generation_systems_tutorial'],
  ['video-generation', '视频生成系统', 'Video Generation', 'generation-systems', 'video_generation_tutorial'],
  ['3d-generation', '3D 生成', '3D Generation', 'generation-systems', '3d_generation_tutorial'],
  ['diffusion-post-training', 'Diffusion 后训练', 'Diffusion Post-Training', 'generation-systems', 'diffusion_post_training_tutorial'],
  ['modern-diffusion-post-training', '现代 Diffusion 后训练', 'Modern Diffusion Post-Training', 'generation-systems', 'modern_diffusion_post_training_tutorial'],
  ['diffusion-distillation', 'Diffusion 与 Flow 蒸馏', 'Diffusion & Flow Distillation', 'generation-systems', 'diffusion_distillation_tutorial'],
  ['vlm-multimodal', '视觉语言模型', 'VLM & Multimodal', 'multimodal', 'vlm_multimodal_tutorial'],
  ['agent-foundations', 'Agent 基础', 'Agent Foundations', 'agents', 'agent_foundations_tutorial'],
  ['agentic-rl', 'Agentic RL', 'Agentic RL', 'agents', 'agentic_rl_tutorial'],
  ['multi-agent-long-horizon', '多智能体与长时程任务', 'Multi-Agent & Long-Horizon', 'agents', 'multi_agent_long_horizon_tutorial'],
  ['self-evolving-agents', '自进化智能体', 'Self-Evolving Agents', 'agents', 'self_evolving_agents_tutorial'],
  ['rag-embedding-retrieval', 'RAG、Embedding 与检索', 'RAG, Embeddings & Retrieval', 'agents', 'rag_embedding_retrieval_tutorial'],
  ['world-models', '世界模型', 'World Models', 'embodied-world-models', 'world_models_tutorial'],
].map(([slug, title, titleEn, category, source]) => ({
  slug, title, titleEn, category, source, kind: 'tutorial',
  zhPath: `tutorials/zh/${slug}.md`,
  enPath: `tutorials/en/${slug}.md`,
}));

const codeByTutorial = {
  attention: ['mha.py', 'axial_attention.py'],
  'transformer-block': ['transformer_block.py'],
  'llm-evaluation': ['llm_eval_metrics.py'],
  'normalization-init': ['normalization.py'],
  'optimizer-lr': ['optimizer_lr_schedule.py'],
  tokenization: ['tokenization.py'],
  'lora-peft': ['lora.py'],
  'linear-sparse-attention': ['linear_sparse_attention.py'],
  'llm-pretraining': ['pretraining_pipeline.py'],
  'llm-inference-serving': ['inference_serving.py'],
  'flow-matching': ['flow_matching.py'],
  'image-generation': ['dit.py', 'mmdit_block.py', 'toy_mmdit_t2i_pipeline.py'],
  'modern-diffusion-post-training': ['diffusion_online_rl.py'],
  'rag-embedding-retrieval': ['rag_embedding.py'],
  'world-models': ['world_models_toy.py'],
};

const embodiedVolumes = [
  ['embodied-basics', '卷一：通识基础', 'Embodied Basics', '01_basics', 58],
  ['embodied-rl', '卷二：强化学习算法', 'RL Algorithms', '02_rl_algo', 53],
  ['embodied-vla-il', '卷三：VLA 与模仿学习', 'VLA & Imitation Learning', '03_vla_il', 77],
  ['embodied-world-sim', '卷四：世界模型与 Sim2Real', 'World Models & Sim2Real', '04_world_sim', 35],
  ['embodied-engineering', '卷五：工程落地', 'Engineering', '05_engineering', 65],
  ['embodied-legged-control', '卷六：腿足控制与遥操作', 'Legged Control & Teleoperation', '06_legged_control', 58],
  ['embodied-perception-nav', '卷七：3D 感知、SLAM 与导航', '3D Perception, SLAM & Navigation', '07_perception_nav', 70],
  ['embodied-coding-system', '卷八：Coding 与系统设计', 'Coding & System Design', '08_coding_systemdesign', 40],
  ['embodied-frontier', '卷九：近期前沿方向', 'Recent Frontiers', '09_frontier', 8],
].map(([slug, title, titleEn, source, questionCount]) => ({
  slug, title, titleEn, source, questionCount, kind: 'question-bank',
  category: 'embodied-world-models', zhPath: `embodied/${slug}.md`,
}));

const blogs = [
  ['cosmos3-mot', 'NVIDIA Cosmos 3：MoT 架构深度导读', 'cosmos3_mot_guide.html'],
  ['continuous-dlm', 'Continuous DLM 综述：表征视角', 'continuous_dlm_representation_perspective.html'],
  ['diffusion-representation-manifold', '扩散、表征与流形', 'diffusion_representation_manifold.html'],
].map(([slug, title, source]) => ({
  slug, title, titleEn: title, source, kind: 'blog', category: 'blogs',
  htmlPath: `blogs/${slug}.html`,
}));

function commitOf(root) {
  return execFileSync('git', ['-C', root, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
}

function cleanText(value) {
  return value
    .replace(/<[^>]+>/g, ' ')
    .replace(/[*_`#]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function extractQuestions(markdown, document) {
  const questions = [];
  const details = /<details[^>]*>\s*<summary>([\s\S]*?)<\/summary>([\s\S]*?)<\/details>/gi;
  for (const match of markdown.matchAll(details)) {
    const prefix = markdown.slice(0, match.index);
    const heading = [...prefix.matchAll(/^###\s+([^\n]+)/gm)].at(-1)?.[1] ?? '';
    const summary = cleanText(match[1]);
    const idMatch = summary.match(/\b(?:Q|F|N|H)\d+\b/i);
    if (!idMatch && document.kind === 'tutorial' && !/？|\?/.test(summary)) continue;
    const difficulty = match[1].match(/\bL([123])\b/i)?.[1] ?? heading.match(/\bL([123])\b/i)?.[1] ?? '';
    const title = summary
      .replace(/^L[123]\s+/i, '')
      .replace(/^(?:🔥×\d+|低频实录[^ ]*|前沿拓展[^ ]*)\s*/i, '')
      .replace(/^(?:Q|F|N|H)\d+[.、·:\s-]*/i, '')
      .trim();
    questions.push({
      id: `${document.slug}-${questions.length + 1}`,
      documentSlug: document.slug,
      documentTitle: document.title,
      category: document.category,
      kind: document.kind,
      difficulty: difficulty ? `L${difficulty}` : '',
      title,
      answer: match[2].trim(),
      href: `/bagu/${document.kind}/${document.slug}`,
    });
  }
  return questions;
}

const pinnedCommits = {
  aris: 'af9926140546a52ea4ef97172e463977d0c8775c',
  embodied: '001e42385d421294066767f286e786e378595f13',
};
const actualCommits = { aris: commitOf(arisRoot), embodied: commitOf(embodiedRoot) };
if (args['allow-unpinned'] !== 'true') {
  for (const key of Object.keys(pinnedCommits)) {
    if (actualCommits[key] !== pinnedCommits[key]) {
      throw new Error(`${key} source must be pinned to ${pinnedCommits[key]}, received ${actualCommits[key]}. Pass --allow-unpinned true to import a reviewed newer revision.`);
    }
  }
}

await rm(outputRoot, { recursive: true, force: true });
for (const folder of ['tutorials/zh', 'tutorials/en', 'code', 'blogs', 'blog-assets', 'embodied', 'licenses']) {
  await mkdir(join(outputRoot, folder), { recursive: true });
}

const allQuestions = [];
for (const tutorial of tutorials) {
  const zhSource = join(arisRoot, 'docs/tutorials', `${tutorial.source}.md`);
  const enSource = join(arisRoot, 'docs/tutorials', `${tutorial.source}_en.md`);
  tutorial.sourceFile = `${tutorial.source}.md`;
  tutorial.sourceFileEn = `${tutorial.source}_en.md`;
  await cp(zhSource, join(outputRoot, tutorial.zhPath));
  await cp(enSource, join(outputRoot, tutorial.enPath));
  tutorial.code = (codeByTutorial[tutorial.slug] ?? []).map(name => `code/${name}`);
  tutorial.questionCount = extractQuestions(await readFile(zhSource, 'utf8'), tutorial).length;
  allQuestions.push(...extractQuestions(await readFile(zhSource, 'utf8'), tutorial));
  delete tutorial.source;
}
for (const names of Object.values(codeByTutorial)) {
  for (const name of names) {
    await cp(join(arisRoot, 'docs/tutorials/code', name), join(outputRoot, 'code', name));
  }
}
for (const volume of embodiedVolumes) {
  const source = join(embodiedRoot, 'docs/interviews', `${volume.source}.md`);
  await cp(source, join(outputRoot, volume.zhPath));
  allQuestions.push(...extractQuestions(await readFile(source, 'utf8'), volume));
  delete volume.source;
}
for (const blog of blogs) {
  const sourcePath = join(arisRoot, 'docs/blogs', blog.source);
  let html = await readFile(sourcePath, 'utf8');
  const remoteImages = [...html.matchAll(/https:\/\/raw\.githubusercontent\.com\/wanshuiyin\/wanshuiyin\.github\.io\/master\/image\/([^"'?#]+)/g)];
  for (const match of remoteImages) {
    const name = match[1];
    const pinnedUrl = `https://raw.githubusercontent.com/wanshuiyin/wanshuiyin.github.io/f135a0232debe5dfa79db705d24e08a8249c70fd/image/${name}`;
    const target = join(outputRoot, 'blog-assets', name);
    try {
      await readFile(target);
    } catch {
      const response = await fetch(pinnedUrl);
      if (!response.ok) throw new Error(`Failed to download blog image ${name}: ${response.status}`);
      await writeFile(target, Buffer.from(await response.arrayBuffer()));
    }
    html = html.replaceAll(match[0], `../blog-assets/${name}`);
  }
  html = html.replaceAll(
    'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js',
    '../vendor/mathjax/tex-mml-chtml.js',
  );
  await writeFile(join(outputRoot, blog.htmlPath), html);
  delete blog.source;
}
await mkdir(join(outputRoot, 'vendor'), { recursive: true });
await mkdir(join(outputRoot, 'vendor/mathjax/output/chtml/fonts'), { recursive: true });
await cp(
  join(mathjaxRoot, 'tex-mml-chtml.js'),
  join(outputRoot, 'vendor/mathjax/tex-mml-chtml.js'),
);
await cp(
  join(mathjaxRoot, 'output/chtml/fonts/woff-v2'),
  join(outputRoot, 'vendor/mathjax/output/chtml/fonts/woff-v2'),
  { recursive: true },
);

await cp(join(arisRoot, 'LICENSE'), join(outputRoot, 'licenses', 'ARIS-in-AI-Offer-LICENSE.txt'));
await cp(join(embodiedRoot, 'LICENSE'), join(outputRoot, 'licenses', 'embodied-interview-qa-LICENSE.txt'));
await cp(join(arisRoot, 'README_CN.md'), join(outputRoot, 'licenses', 'ARIS-in-AI-Offer-README_CN.md'));
await cp(join(embodiedRoot, 'README.md'), join(outputRoot, 'licenses', 'embodied-interview-qa-README.md'));

const sources = [
  {
    id: 'aris-ai-offer',
    name: 'ARIS-in-AI-Offer',
    author: 'Ruofeng Yang (杨若峰)',
    url: 'https://github.com/wanshuiyin/ARIS-in-AI-Offer',
    commit: actualCommits.aris,
    license: 'MIT',
  },
  {
    id: 'embodied-interview-qa',
    name: 'embodied-interview-qa',
    author: 'WinstonJQ',
    url: 'https://github.com/WinstonJQ/embodied-interview-qa',
    commit: actualCommits.embodied,
    license: 'MIT',
  },
];
const documents = [...tutorials, ...embodiedVolumes, ...blogs];
if (tutorials.length !== 35 || blogs.length !== 3 || embodiedVolumes.length !== 9) {
  throw new Error('Handbook catalog counts are incomplete.');
}
if (new Set(documents.map((document) => `${document.kind}:${document.slug}`)).size !== documents.length) {
  throw new Error('Handbook document slugs must be unique within each collection.');
}
if (allQuestions.filter((question) => question.kind === 'question-bank').length !== 464) {
  throw new Error('Embodied question import must contain exactly 464 entries for the pinned revision.');
}
await writeFile(join(outputRoot, 'index.json'), `${JSON.stringify({
  version: 2,
  importedAt: new Date().toISOString(),
  title: 'Ember AI 理论手册',
  subtitle: '从基础原理到大模型系统、生成模型、智能体与具身智能',
  categories,
  documents,
  sources,
  stats: {
    tutorials: tutorials.length,
    languages: 2,
    codeExamples: new Set(Object.values(codeByTutorial).flat()).size,
    blogs: blogs.length,
    embodiedVolumes: embodiedVolumes.length,
    embodiedQuestions: embodiedVolumes.reduce((sum, volume) => sum + volume.questionCount, 0),
    interviewQuestions: allQuestions.length,
  },
}, null, 2)}\n`);
await writeFile(join(outputRoot, 'questions.json'), `${JSON.stringify({ questions: allQuestions }, null, 2)}\n`);
await writeFile(join(outputRoot, 'NOTICE.md'), `# Third-party notices

The handbook content is vendored for offline reading and remains attributed to its original authors.

## ARIS-in-AI-Offer

- Copyright (c) 2026 Ruofeng Yang (杨若峰)
- Source: https://github.com/wanshuiyin/ARIS-in-AI-Offer
- Commit: ${sources[0].commit}
- License: MIT (see licenses/ARIS-in-AI-Offer-LICENSE.txt)

## embodied-interview-qa

- Copyright (c) 2026 WinstonJQ
- Source: https://github.com/WinstonJQ/embodied-interview-qa
- Commit: ${sources[1].commit}
- License: MIT (see licenses/embodied-interview-qa-LICENSE.txt)

Technical blog figures may retain additional third-party copyrights identified in the articles. Ember does not claim authorship of vendored material.
Remote figures referenced by the diffusion representation blog are mirrored from wanshuiyin/wanshuiyin.github.io at commit f135a0232debe5dfa79db705d24e08a8249c70fd so the handbook remains readable offline.
`);

console.log(`Imported ${documents.length} documents and ${allQuestions.length} interview questions to ${outputRoot}`);
