# Ember 文档

文档站使用 Fern，正文位于 [`fern/pages`](fern/pages)，导航位于 [`fern/index.yml`](fern/index.yml)，外观与实例配置位于 [`fern/docs.yml`](fern/docs.yml)。

需要 Node.js 22 或更新版本。在仓库根目录运行：

```bash
npm run docs:install
npm run docs:check
npm run docs:dev
```

打开 <http://localhost:4310>。第一次预览会下载 Fern 运行时。停止时按 Ctrl+C。文档依赖独立安装，不进入桌面应用。

本站按任务组织为五个入口：首页、桌面使用、开发指南、参考、贡献。中文正文使用英文 kebab-case 文件名与稳定 slug；教程使用 Steps，环境分支使用 Tabs，纯说明采用 Markdown。详细规则见[写作规范](fern/pages/community/contributing/documentation/style-guide.md)，新增页面可从 [`fern/templates`](fern/templates) 开始。

`npm run docs:check` 检查导航覆盖、文件名、frontmatter、标题、语言围栏、相对文件与标题链接、本地源码引用、Fern 配置及 MDX。GitHub Actions 运行同一检查，不自动发布。

当前 `ember-local` 组织与实例是本地预览配置，不代表已上线站点。正式发布前替换为维护者拥有的 Fern 组织和实例，完成认证，再按[发布说明](fern/pages/community/contributing/documentation/local-preview.mdx)操作。

站点组织参考了 [Dynamo 的 Fern 文档](https://github.com/ai-dynamo/dynamo/tree/main/docs/fern)，内容根据 Ember 当前实现编写。内部设计与性能原始记录继续保留在被 Git 忽略的项目 `wiki`；公开站点不依赖该目录。
