# Ember 文档维护

修改正文前阅读 `pages/community/contributing/documentation/style-guide.md`。

- 以当前实现核对命令、默认值和兼容边界；不要虚构公开下载渠道。
- 页面正文使用中文；文件名与URL slug使用英文kebab-case。
- 每页提供title/subtitle frontmatter，正文从H2开始。
- 新增页面同时维护index.yml，使用相对页面链接；源码用本仓库GitHub绝对URL。
- 内部方案与原始实验记录放项目wiki，公开正文不引用ignored文件。
- 从仓库根目录执行 `npm run docs:check`；涉及布局或组件时启动 `npm run docs:dev` 检查真实页面。
- Fern组织和实例当前仅用于本地预览。校验与预览不发布文档。
