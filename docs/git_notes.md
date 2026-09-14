# Git 操作笔记

## 1. 三个区域

```text
工作区（Working Tree）
    ↓ git add
暂存区（Index / Staging Area）
    ↓ git commit
本地仓库（Local Repository）
    ↓ git push
GitHub（Remote Repository）
```

- 工作区：正在编辑或生成的普通文件。
- 暂存区：已经挑选好、准备进入下一次 commit 的内容。
- 本地仓库：已经形成的 commit 历史。
- GitHub：远程仓库，只有 push 后才会同步。

## 2. 核心命令

### `git status`

查看当前状态：

```powershell
git status
git status --short
```

短状态示例：

```text
?? file.txt       未跟踪
A  file.txt       新文件已暂存
M  file.txt       修改已暂存
 M file.txt       修改未暂存
AM file.txt       已暂存后又在工作区修改
```

### `git diff`

查看工作区和暂存区的差异：

```powershell
git diff
git diff -- requirements.txt
```

如果文件还是 `??` 未跟踪状态，`git diff` 默认不会显示它。

### `git add`

把选定内容放入暂存区：

```powershell
git add -- requirements.txt
git add -- README.md docs/
```

`git add` 不会创建 commit。

### `git diff --cached`

查看暂存区和上一次 commit 的差异：

```powershell
git diff --cached
git diff --cached -- requirements.txt
```

### `git commit`

把暂存区保存为本地历史：

```powershell
git commit -m "Organize CGCNN reproduction and multitask experiment"
```

执行 commit 前必须先确认：

- `git status` 中要提交的文件是否正确；
- `git diff --cached` 中没有密码、Token、API Key；
- 没有把大数据和临时 checkpoint 加进去；
- commit message 能说明这次保存了什么。

### `git log`

查看历史：

```powershell
git log --oneline --decorate
git log --oneline --decorate --graph --all
```

### `git remote`

查看远程仓库：

```powershell
git remote -v
```

添加 GitHub 地址：

```powershell
git remote add origin <GitHub repository URL>
```

`origin` 只是远程仓库的本地别名。

### `git push`

把本地 commit 上传到 GitHub：

```powershell
git push -u origin main
```

`-u` 会记住上游分支，以后可以直接运行 `git push`。

本项目要求真正执行 push 前停下来确认，不能自动输入 GitHub 密码或 Token。

## 3. 当前项目的历史状态

检查时：

- 项目已有 `.git/`；
- 当前分支是 `master`；
- 还没有任何 commit；
- `git-practice.txt` 已经被暂存，但暂存后又被修改；
- 其他项目文件最初都是未跟踪文件；
- 没有配置 GitHub remote。

这意味着项目还没有一个可回溯的 Git 基线。

## 4. 已完成的 Git 教学操作

对 `requirements.txt` 实际执行过：

```powershell
git diff -- requirements.txt
git add -- requirements.txt
git diff --cached -- requirements.txt
git status --short -- requirements.txt
```

结果：

- `git diff` 为空，因为文件尚未跟踪；
- `git add` 后状态变为 `A`；
- `git diff --cached` 显示了准备提交的完整内容。

## 5. .gitignore 的作用

`.gitignore` 只阻止未跟踪文件进入 Git，不会删除文件。

本项目当前策略：

- 保留小样本数据；
- 忽略大型数据集；
- 忽略用户生成的 checkpoint；
- 保留官方 `pre-trained/` 小模型；
- 保留 `results/` 的真实日志和指标；
- 忽略 IDE、缓存和临时日志。

检查某个路径是否被忽略：

```powershell
git check-ignore -v -- data/mt-large/id_prop.csv
```

## 6. 推荐提交顺序

为了历史清楚，建议至少分成：

1. 保存当前 CGCNN 代码和项目结构；
2. 保存 README、论文笔记和汇报材料；
3. 保存真实实验结果和图表。

如果只提交一次，也要在 commit 前完整检查 `git diff --cached`。

## 7. 不要做的事情

- 不要用 `git reset --hard` 清理不理解的改动；
- 不要在没有备份时删除 `.git/`；
- 不要提交 Token、密码、API Key；
- 不要 `git add .` 后不看暂存内容就 commit；
- 不要自动 push。

## 8. 一句话记忆

`git status` 看位置，`git diff` 看未暂存改动，`git add` 挑选内容，`git diff --cached` 确认提交内容，`git commit` 保存本地历史，`git remote` 配置 GitHub，`git push` 上传。
